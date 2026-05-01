# Softplusmax — drop-in for softmax

Compare `softplusmax(z)_i = softplus(z_i) / Σ_j softplus(z_j)` against
the standard softmax across three CIFAR-100 / ResNet-56 tasks:

  | Task | Model | Loss site |
  |---|---|---|
  | `cls`     | ResNet-56          | output cross-entropy |
  | `kd`      | ResNet-20 (teacher: ResNet-56) | KD soft target |
  | `infonce` | ResNet-56 + 2-layer MLP head | SimCLR contrastive |

## Quick smoke test

```bash
python losses.py                  # all 6 losses + backward
python -m models.resnet_cifar     # resnet20 / resnet56 forward
```

## Run a single experiment (single seed, ~hours on 1 GPU)

```bash
# Stage A: classification
python train.py --task cls --loss sm
python train.py --task cls --loss sp

# Stage B: distillation (uses cls best.pt as teacher)
python train.py --task kd --loss sm --T 4 --teacher_ckpt runs/cls_sm/best.pt
python train.py --task kd --loss sm --T 1 --teacher_ckpt runs/cls_sm/best.pt
python train.py --task kd --loss sm --T 8 --teacher_ckpt runs/cls_sm/best.pt
python train.py --task kd --loss sp     --teacher_ckpt runs/cls_sm/best.pt

# Stage C: InfoNCE pre-training, then linear eval
python train.py --task infonce --loss sm --tau 0.1
python eval.py  --ckpt runs/infonce_sm_tau0.1/last.pt

python train.py --task infonce --loss sp --tau 0.1
python eval.py  --ckpt runs/infonce_sp_tau0.1/last.pt
```

## τ sweep (Stage C — main contribution)

```bash
for tau in 0.05 0.1 0.2 0.5 1.0; do
    python train.py --task infonce --loss sm --tau "$tau"
    python eval.py  --ckpt "runs/infonce_sm_tau${tau}/last.pt"

    python train.py --task infonce --loss sp --tau "$tau"
    python eval.py  --ckpt "runs/infonce_sp_tau${tau}/last.pt"
done
```

## Layout

```
losses.py                 SoftmaxCE / SoftplusmaxNLL / SoftmaxKD / …
models/
  resnet_cifar.py         CIFAR ResNet-20 / ResNet-56 (He 2016)
  proj_head.py            SimCLR projection head
train.py                  one loop, three tasks
eval.py                   linear eval for InfoNCE backbones
```

## Defaults (single seed)

- 200 epoch (cls / kd), 500 epoch (infonce)
- SGD-Nesterov, momentum 0.9, weight_decay 5e-4
- Cosine LR with 5-epoch linear warmup, base LR 0.1
- Batch 128 (cls / kd), 512 (infonce)
- CIFAR-100 standard augmentation
