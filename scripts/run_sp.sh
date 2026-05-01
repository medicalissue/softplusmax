#!/usr/bin/env bash
# Softplusmax pipeline: cls_sp → kd_sp (teacher=cls_sp, no T)
#                              → infonce_sp (no τ) + linear eval.
# Same-family teacher (sp ← sp) for distillation. Self-terminates at end.
set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace}"
S3="${S3_PREFIX:-s3://nelu-checkpoints/softplusmax}"
cd "$REPO_DIR"

WB_FLAG="--wandb"

echo "════════════════ Stage A: cls_sp ════════════════"
python train.py --task cls --loss sp $WB_FLAG
aws s3 sync runs/cls_sp/ "$S3/cls_sp/" --quiet

echo "════════════════ Stage B: kd_sp (teacher = cls_sp, no T) ════════════════"
python train.py --task kd --loss sp \
    --teacher_ckpt runs/cls_sp/best.pt $WB_FLAG
aws s3 sync runs/kd_sp/ "$S3/kd_sp/" --quiet

echo "════════════════ Stage C: infonce_sp (no τ) ════════════════"
python train.py --task infonce --loss sp \
    --batch_size 512 --epochs 500 $WB_FLAG
aws s3 sync runs/infonce_sp/ "$S3/infonce_sp/" --quiet

echo "════════════════ Linear eval (sp) ════════════════"
python eval.py --ckpt runs/infonce_sp/last.pt $WB_FLAG
aws s3 sync runs/infonce_sp/linear_eval/ \
    "$S3/infonce_sp/linear_eval/" --quiet

echo "════════════════ DONE (sp pipeline) ════════════════"
date -u +%FT%TZ | aws s3 cp - "$S3/sp_complete"
