#!/usr/bin/env bash
# Softplusmax InfoNCE only (τ=0.5, sm-mirror) + linear eval.
# Self-terminates at end.
set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace}"
S3="${S3_PREFIX:-s3://nelu-checkpoints/softplusmax}"
cd "$REPO_DIR"

WB_FLAG="--wandb"

echo "════════════════ infonce_sp τ=0.5 ════════════════"
python train.py --task infonce --loss sp --tau 0.5 \
    --batch_size 512 --epochs 500 $WB_FLAG
aws s3 sync runs/infonce_sp_tau0.5/ "$S3/infonce_sp_tau0.5/" --quiet

echo "════════════════ Linear eval (sp) ════════════════"
python eval.py --ckpt runs/infonce_sp_tau0.5/last.pt $WB_FLAG
aws s3 sync runs/infonce_sp_tau0.5/linear_eval/ \
    "$S3/infonce_sp_tau0.5/linear_eval/" --quiet

echo "════════════════ DONE (sp infonce τ=0.5) ════════════════"
date -u +%FT%TZ | aws s3 cp - "$S3/sp_infonce_tau05_complete"
