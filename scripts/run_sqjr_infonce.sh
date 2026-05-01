#!/usr/bin/env bash
# SqJumpReLU InfoNCE: hyperparameter-free, θ=tanh(raw), θ_init=0 learnable.
# No τ — SqJR is scale-invariant.
set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace}"
S3="${S3_PREFIX:-s3://nelu-checkpoints/softplusmax}"
cd "$REPO_DIR"

WB_FLAG="--wandb"

echo "════════════════ infonce_sqjr θ=0 learnable (no τ) ════════════════"
python train.py --task infonce --loss sqjr \
    --sqjr_theta_init 0.0 --sqjr_theta_learnable \
    --batch_size 512 --epochs 500 $WB_FLAG
aws s3 sync runs/infonce_sqjr_th0L/ "$S3/infonce_sqjr_th0L/" --quiet

echo "════════════════ Linear eval (sqjr) ════════════════"
python eval.py --ckpt runs/infonce_sqjr_th0L/last.pt $WB_FLAG
aws s3 sync runs/infonce_sqjr_th0L/linear_eval/ \
    "$S3/infonce_sqjr_th0L/linear_eval/" --quiet

echo "════════════════ DONE ════════════════"
date -u +%FT%TZ | aws s3 cp - "$S3/sqjr_infonce_complete"
