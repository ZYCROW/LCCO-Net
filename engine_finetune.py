# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# LCCO-Net training / evaluation engine.
#
# Implements the Paired Objective Co-Optimization (Section 3.7):
#   L = L_CE + lambda * L_CSDD,
# where L_CSDD = beta1 * L_FAD + beta2 * L_CCO (Cross-Scale Difference Digging).
# L_FAD is the class-conditional Feature Alignment and Discrepancy loss with a
# sample-adaptive margin driven by the LHFE high-frequency energy; L_CCO is the
# class-conditional Orthogonal (Complementary) constraint on deep features.

import math
from typing import Iterable, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from timm.data import Mixup
from timm.utils import accuracy, ModelEma
from scipy.special import softmax
from sklearn.metrics import average_precision_score, accuracy_score

import utils
from utils import adjust_learning_rate
from models.cco import cco_loss


def _fad_loss(feat_a, feat_b, margins, targets, eps=1e-8):
    """Feature Alignment and Discrepancy (FAD) loss (Section 3.5).

    Real samples (y=0): minimize the normalized squared distance (alignment).
    Fake samples (y=1): hinge loss requiring the normalized distance to exceed
    the sample-adaptive margin.

    Args:
        feat_a, feat_b: [B, D] features from the two branches.
        margins: [B] sample-adaptive discrepancy margins.
        targets: [B] labels, 0 = real, 1 = fake.
    Returns:
        loss: scalar FAD loss.
        real_dist_mean: mean normalized squared distance of real samples, used
            to update the EMA baseline of the adaptive margin.
    """
    a_norm = F.normalize(feat_a, p=2, dim=1, eps=eps)
    b_norm = F.normalize(feat_b, p=2, dim=1, eps=eps)
    dist_sq = (a_norm - b_norm).pow(2).sum(dim=1)  # [B]

    real_mask = (targets == 0).float()
    fake_mask = (targets == 1).float()

    real_loss = (dist_sq * real_mask).sum() / (real_mask.sum() + eps)
    fake_loss = (F.relu(margins - dist_sq) * fake_mask).sum() / (fake_mask.sum() + eps)

    if real_mask.sum() > 0:
        real_dist_mean = (dist_sq * real_mask).sum() / real_mask.sum()
    else:
        real_dist_mean = margins.mean()

    return real_loss + fake_loss, real_dist_mean


def compute_losses(model, samples, targets, criterion, args):
    """Forward pass and joint loss computation (CE + lambda * CSDD)."""
    (logits, deep_u, deep_l, shallow_u, shallow_l,
     hf_energy, margins) = model(samples)

    # Classification objective.
    ce_loss = criterion(logits, targets)

    # FAD at shallow and deep levels (Section 3.5).
    shallow_fad, shallow_real_mean = _fad_loss(
        shallow_u, shallow_l, margins, targets)
    deep_fad, deep_real_mean = _fad_loss(
        deep_u, deep_l, margins, targets)

    # Cross-scale FAD with learnable scale-balancing coefficients (gamma1, gamma2).
    m = model.module if hasattr(model, 'module') else model
    w_deep, w_shallow = m.get_scale_weights()
    fad_loss = w_deep * deep_fad + w_shallow * shallow_fad

    # Update the EMA of the natural real-sample discrepancy (margin baseline).
    if model.training:
        cross_real_mean = 0.5 * (deep_real_mean + shallow_real_mean)
        m.update_real_dist_ema(cross_real_mean)

    # Class-conditional Orthogonal constraint on deep features (Section 3.4).
    cco = cco_loss(deep_u, deep_l, targets)

    # Cross-Scale Difference Digging (CSDD, Section 3.6).
    csdd_loss = args.fad_beta * fad_loss + args.cco_beta * cco

    # Paired Objective Co-Optimization: L = CE + lambda * CSDD (lambda=0.5).
    loss = ce_loss + args.csdd_weight * csdd_loss

    return {
        'loss': loss,
        'output': logits,
        'ce_loss': ce_loss,
        'fad_loss': fad_loss,
        'deep_fad': deep_fad,
        'shallow_fad': shallow_fad,
        'cco_loss': cco,
        'csdd_loss': csdd_loss,
        'w_deep': w_deep,
        'w_shallow': w_shallow,
        'margin_mean': margins.mean(),
        'hf_energy_mean': hf_energy.mean(),
        'deep_real_mean': deep_real_mean,
        'shallow_real_mean': shallow_real_mean,
    }


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    log_writer=None, args=None):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    update_freq = args.update_freq
    use_amp = args.use_amp
    optimizer.zero_grad()

    for data_iter_step, (samples, targets) in enumerate(
            metric_logger.log_every(data_loader, print_freq=100, header=header)):

        # Per-iteration learning rate schedule.
        if data_iter_step % update_freq == 0:
            adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if use_amp:
            with torch.cuda.amp.autocast():
                L = compute_losses(model, samples, targets, criterion, args)
        else:
            L = compute_losses(model, samples, targets, criterion, args)

        loss = L['loss']
        loss_value = loss.item()

        # Log all loss components and auxiliary statistics.
        metric_logger.update(ce_loss=L['ce_loss'].item())
        metric_logger.update(fad_loss=L['fad_loss'].item())
        metric_logger.update(deep_fad=L['deep_fad'].item())
        metric_logger.update(shallow_fad=L['shallow_fad'].item())
        metric_logger.update(cco_loss=L['cco_loss'].item())
        metric_logger.update(csdd_loss=L['csdd_loss'].item())
        metric_logger.update(w_deep=L['w_deep'].item())
        metric_logger.update(w_shallow=L['w_shallow'].item())
        metric_logger.update(margin_mean=L['margin_mean'].item())
        metric_logger.update(hf_energy_mean=L['hf_energy_mean'].item())
        metric_logger.update(deep_real_dist=L['deep_real_mean'].item())
        metric_logger.update(shallow_real_dist=L['shallow_real_mean'].item())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            assert math.isfinite(loss_value)

        if use_amp:
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
        else:
            loss /= update_freq
            loss.backward()
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.step()
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)

        torch.cuda.synchronize()

        output = L['output']
        if mixup_fn is None:
            class_acc = (output.max(-1)[-1] == targets).float().mean()
        else:
            class_acc = None

        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])
        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        if use_amp:
            metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            if use_amp:
                log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.update(ce_loss=L['ce_loss'].item(), head="loss")
            log_writer.update(fad_loss=L['fad_loss'].item(), head="loss")
            log_writer.update(deep_fad=L['deep_fad'].item(), head="loss")
            log_writer.update(shallow_fad=L['shallow_fad'].item(), head="loss")
            log_writer.update(cco_loss=L['cco_loss'].item(), head="loss")
            log_writer.update(csdd_loss=L['csdd_loss'].item(), head="loss")
            log_writer.update(w_deep=L['w_deep'].item(), head="weights")
            log_writer.update(w_shallow=L['w_shallow'].item(), head="weights")
            log_writer.update(margin_mean=L['margin_mean'].item(), head="margins")
            log_writer.update(hf_energy_mean=L['hf_energy_mean'].item(), head="margins")
            log_writer.set_step()

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, val=None, use_amp=False):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    model.eval()

    for index, batch in enumerate(metric_logger.log_every(data_loader, 1000, header)):
        images = batch[0]
        target = batch[-1]

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(images)
                if isinstance(output, tuple):
                    output = output[0]
                loss = criterion(output, target)
        else:
            output = model(images)
            if isinstance(output, tuple):
                output = output[0]
            loss = criterion(output, target)

        if index == 0:
            predictions = output
            labels = target
        else:
            predictions = torch.cat((predictions, output), 0)
            labels = torch.cat((labels, target), 0)

        torch.cuda.synchronize()

        acc1, _ = [acc / 100 for acc in accuracy(output, target, topk=(1, 2))]

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)

    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.2%} loss {losses.global_avg:.4f}'
          .format(top1=metric_logger.acc1, losses=metric_logger.loss))

    output_ddp = [torch.zeros_like(predictions) for _ in range(utils.get_world_size())]
    dist.all_gather(output_ddp, predictions)
    labels_ddp = [torch.zeros_like(labels) for _ in range(utils.get_world_size())]
    dist.all_gather(labels_ddp, labels)

    output_all = torch.concat(output_ddp, dim=0)
    labels_all = torch.concat(labels_ddp, dim=0)

    y_pred = softmax(output_all.detach().cpu().numpy(), axis=1)[:, 1]
    y_true = labels_all.detach().cpu().numpy().astype(int)

    acc = accuracy_score(y_true, y_pred > 0.5)
    ap = average_precision_score(y_true, y_pred)

    y_pred_label = (y_pred > 0.5).astype(int)
    real_mask = (y_true == 0)
    fake_mask = (y_true == 1)
    realacc = (y_pred_label[real_mask] == 0).mean() if real_mask.sum() > 0 else float('nan')
    fakeacc = (y_pred_label[fake_mask] == 1).mean() if fake_mask.sum() > 0 else float('nan')

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, acc, ap, realacc, fakeacc
