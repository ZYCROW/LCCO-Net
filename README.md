# LCCO-Net

**LCCO-Net: A Layer-twice Cross-scale Complementary Optimization Network for AI-Generated Image Detection**

LCCO-Net is a difference-guided and frequency-adaptive dual-layer (dual-branch)
framework for detecting AI-generated images. It is built on the observation that
synthetic images contain both authentic patterns and synthetic traces (a bimodal
distribution), whereas real images follow a unimodal distribution. LCCO-Net
explicitly amplifies this internal inconsistency through three complementary,
lightweight, plug-and-play modules:

1. **LHFE (Local High-Frequency Enhancement)**: placed at the input end and
   shared by both branches. It decomposes the input with a learnable blur kernel,
   extracts the high-frequency residual `H = X - LowPass(X)`, and enhances the
   input as `X' = X + alpha * H` (alpha = 1). The per-sample high-frequency
   energy `E = mean(|H|)` is returned and used to build the sample-adaptive
   discrepancy margin.
2. **Difference-guided CBAM**: one CBAM is applied at the shallow (Process 1) and
   deep (Process 2) stages. The attention map is derived from the inter-branch
   discrepancy map `|f_upper - f_lower|` and then used to modulate the features
   of both branches, emphasizing forgery-relevant channels and spatial regions.
3. **CCO (Class-conditional Orthogonal / Complementary constraint)**: imposed on
   the deep features of the two branches. For real samples it minimizes the
   squared Euclidean distance (consistency); for fake samples it minimizes the
   squared Frobenius norm of the cross-inner-product matrix (low correlation /
   complementarity).

These modules cooperate as **enhance -> focus -> diversify**: LHFE supplies
high-frequency cues, CBAM selects the most diagnostic channels and locations, and
CCO prevents the two branches from collapsing into similar feature spaces.

## Method Pipeline

```
Input X (Real / Fake)
      |
      v
  [ LHFE ]  -->  enhanced X'  and  per-sample high-frequency energy E
      |
      +--> Branch Upper -- Process 1 (shallow) --> f_u,s -- [diff-guided CBAM]
      |                 \-- Process 2 (deep)   --> f_u,d -- [diff-guided CBAM]
      |
      +--> Branch Lower -- Process 1 (shallow) --> f_l,s -- [diff-guided CBAM]
                        \-- Process 2 (deep)   --> f_l,d -- [diff-guided CBAM]

  Shallow f_u,s / f_l,s  -->  FAD loss (sample-adaptive margin from E)
  Deep    f_u,d / f_l,d  -->  FAD loss (sample-adaptive margin from E)
  Deep    f_u,d / f_l,d  -->  CCO loss (class-conditional)
  FAD + CCO              -->  CSDD loss (Cross-Scale Difference Digging)
  Deep (f_u,d - f_l,d)   -->  Classifier --> Real / Fake  (CE loss)

  Total loss:  L = L_CE + lambda * L_CSDD,   lambda = 0.5
  L_CSDD = beta1 * L_FAD + beta2 * L_CCO
```

The adaptive margin for fake samples is
`M(x) = real_dist_ema + margin_coeff * normalized_E(x)`,
where `real_dist_ema` is the EMA of the natural cross-scale discrepancy of real
samples in the current batch, and `normalized_E` is the min-max normalized
high-frequency energy within the batch. Samples with stronger high-frequency
anomalies are required to exhibit a larger inter-branch discrepancy.

## Directory Structure

```
LCCO-Net/
├── checkpoint/            # saved / fine-tuned weights (.pth)
├── data/
│   ├── datasets.py        # TrainDataset: loading and train/test augmentation
│   └── generation.py      # optional: generate fake samples with diffusers
├── models/
│   ├── lcco_net.py        # TwoBranchLCCONet (main model, ResNet / ConvNeXt)
│   ├── lhfe.py            # Local High-Frequency Enhancement module
│   ├── cbam.py            # Convolutional Block Attention Module
│   ├── cco.py             # Class-conditional Orthogonal constraint loss
│   ├── resnet.py          # ResNet building blocks and single-branch baseline
│   └── convnext.py        # ConvNeXt-tiny residual backbone (self-contained)
├── scripts/
│   ├── train.sh           # distributed training entry
│   └── eval.sh            # inference / evaluation entry
├── engine_finetune.py     # training/evaluation engine (CE + FAD + CCO -> CSDD)
├── main_finetune.py       # main entry: arguments, model build, orchestration
├── optim_factory.py       # optimizer factory and layer-wise LR decay
├── utils.py               # distributed, logging, AMP, checkpoint utilities
├── requirements.txt
└── README.md
```

## Environment Setup

```bash
conda create -n lcco python=3.9 -y
conda activate lcco
pip install torch==2.2.1 torchvision==0.17.1
pip install -r requirements.txt
```

## Datasets

LCCO-Net is evaluated on two cross-generator benchmarks:

- **AIGCDetectBenchmark**: trained on ProGAN, tested on 8 generators
  (ProGAN, BigGAN, CycleGAN, ADM, WFIR, SDv-1.4, SDv-1.5, Midjourney).
- **GenImage**: trained on real ImageNet images + SDv1.4 fakes, tested on 7
  generators (SDv1.4, SDv1.5, ADM, BigGAN, Wukong, VQDM, Midjourney).

Each dataset folder follows the `0_real` / `1_fake` sub-directory convention:

```
dataset_root/
├── 0_real/    # real images
└── 1_fake/    # AI-generated images
```

## Training

```bash
bash scripts/train.sh
```

Key hyper-parameters (matching the paper):

| Parameter | Value | Description |
|---|---|---|
| `--batch_size` | 32 | per-GPU batch size |
| `--epochs` | 100 | training epochs |
| `--opt` | adamw | optimizer |
| `--weight_decay` | 0.01 | weight decay |
| `--input_size` | 256 | image input size |
| `--backbone` | resnet / convnext | branch backbone (ResNet or ConvNeXt-tiny) |
| `--alpha` | 1.0 | LHFE high-frequency amplification |
| `--csdd_weight` | 0.5 | lambda, weight of CSDD in total loss |
| `--fad_beta` | 0.5 | beta1, FAD weight inside CSDD |
| `--cco_beta` | 0.5 | beta2, CCO weight inside CSDD |
| `--margin_coeff` | 0.30 | coefficient of high-frequency energy in adaptive margin |

Data augmentation follows the paper: RandomCrop, RandomHorizontalFlip,
RandomRotation, and RandomMask.

## Inference / Evaluation

Place the checkpoint under `./checkpoint/`, then:

```bash
bash scripts/eval.sh
```

The evaluation script iterates over all generator sub-directories under the
given benchmark root and reports per-generator accuracy, average precision,
real accuracy, fake accuracy, and the mean accuracy.

## Main Results

Average accuracy (%) on the two benchmarks (higher is better):

| Benchmark | Backbone | LCCO-Net | Strongest baseline |
|---|---|---|---|
| AIGCDetectBenchmark | ResNet | **94.56** | AIDE 93.43 |
| AIGCDetectBenchmark | ConvNeXt | **95.55** | AIDE 93.43 |
| GenImage | ResNet | **93.58** | DADL 91.98 |
| GenImage | ConvNeXt | **93.90** | DADL 91.98 |

Ablation (AIGCDetectBenchmark, ResNet backbone):

| Configuration | Avg Acc | Drop |
|---|---|---|
| Full LCCO-Net | 94.56 | - |
| w/o LHFE | 81.80 | -12.76 |
| w/o CBAM | 88.20 | -6.36 |
| w/o CCO | 92.58 | -1.98 |

The contribution ranking is **LHFE > CBAM > CCO**, confirming that high-frequency
enhancement is the most critical component for cross-generator generalization.

## Notes

- Label convention: `0 = real`, `1 = fake`.
- The CBAM at each stage shares weights between the two branches and is applied
  to each branch separately (difference-guided).
- The shallow / deep FAD losses use learnable scale-balancing coefficients
  (softmax-normalized, summing to 1).
- The adaptive margin combines the EMA of real-sample natural discrepancy with
  the per-sample high-frequency energy, so it is not a fixed hyper-parameter.
