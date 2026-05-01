#!/usr/bin/env bash
# Softmax pipeline: cls_sm → kd_sm_T4 (teacher=cls_sm) → infonce_sm + linear eval.
# Uploads each run to S3 immediately on completion. Self-terminates at the end.
set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace}"
S3="${S3_PREFIX:-s3://nelu-checkpoints/softplusmax}"
cd "$REPO_DIR"

WB_FLAG="--wandb"

echo "════════════════ Stage A: cls_sm ════════════════"
python train.py --task cls --loss sm $WB_FLAG
aws s3 sync runs/cls_sm/ "$S3/cls_sm/" --quiet

echo "════════════════ Stage B: kd_sm_T4 ════════════════"
python train.py --task kd --loss sm --T 4 \
    --teacher_ckpt runs/cls_sm/best.pt $WB_FLAG
aws s3 sync runs/kd_sm_T4/ "$S3/kd_sm_T4/" --quiet

echo "════════════════ Stage C: infonce_sm_tau0.5 ════════════════"
python train.py --task infonce --loss sm --tau 0.5 \
    --batch_size 512 --epochs 500 $WB_FLAG
aws s3 sync runs/infonce_sm_tau0.5/ "$S3/infonce_sm_tau0.5/" --quiet

echo "════════════════ Linear eval (sm) ════════════════"
python eval.py --ckpt runs/infonce_sm_tau0.5/last.pt $WB_FLAG
aws s3 sync runs/infonce_sm_tau0.5/linear_eval/ \
    "$S3/infonce_sm_tau0.5/linear_eval/" --quiet

echo "════════════════ DONE (sm pipeline) ════════════════"
date -u +%FT%TZ | aws s3 cp - "$S3/sm_complete"
