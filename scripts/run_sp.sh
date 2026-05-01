#!/usr/bin/env bash
# Softplusmax pipeline: cls_sp → kd_sp_T4 (teacher=cls_sp)
#                              → infonce_sp_tau0.5 + linear eval.
# Same hyperparameters as sm pipeline so the only diff is the function
# (exp ↔ softplus). Self-terminates at end.
set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace}"
S3="${S3_PREFIX:-s3://nelu-checkpoints/softplusmax}"
cd "$REPO_DIR"

WB_FLAG="--wandb"

echo "════════════════ Stage A: cls_sp ════════════════"
python train.py --task cls --loss sp $WB_FLAG
aws s3 sync runs/cls_sp/ "$S3/cls_sp/" --quiet

echo "════════════════ Stage B: kd_sp T=4 (teacher = cls_sp) ════════════════"
python train.py --task kd --loss sp --T 4 \
    --teacher_ckpt runs/cls_sp/best.pt $WB_FLAG
aws s3 sync runs/kd_sp_T4/ "$S3/kd_sp_T4/" --quiet

echo "════════════════ Stage C: infonce_sp τ=0.5 ════════════════"
python train.py --task infonce --loss sp --tau 0.5 \
    --batch_size 512 --epochs 500 $WB_FLAG
aws s3 sync runs/infonce_sp_tau0.5/ "$S3/infonce_sp_tau0.5/" --quiet

echo "════════════════ Linear eval (sp) ════════════════"
python eval.py --ckpt runs/infonce_sp_tau0.5/last.pt $WB_FLAG
aws s3 sync runs/infonce_sp_tau0.5/linear_eval/ \
    "$S3/infonce_sp_tau0.5/linear_eval/" --quiet

echo "════════════════ DONE (sp pipeline) ════════════════"
date -u +%FT%TZ | aws s3 cp - "$S3/sp_complete"
