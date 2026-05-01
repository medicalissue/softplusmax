#!/usr/bin/env bash
# Softplusmax KD with T=4 (z/T scaling + T² correction, sm-mirror).
# Reuses cls_sp teacher from S3. Skips infonce. Self-terminates at end.
set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace}"
S3="${S3_PREFIX:-s3://nelu-checkpoints/softplusmax}"
cd "$REPO_DIR"

WB_FLAG="--wandb"

echo "════════════════ Pull cls_sp teacher from S3 ════════════════"
mkdir -p runs/cls_sp
aws s3 cp "$S3/cls_sp/best.pt" runs/cls_sp/best.pt --quiet
ls -la runs/cls_sp/

echo "════════════════ kd_sp_T4 (teacher = cls_sp, T=4) ════════════════"
python train.py --task kd --loss sp \
    --T 4 \
    --teacher_ckpt runs/cls_sp/best.pt $WB_FLAG
aws s3 sync runs/kd_sp_T4/ "$S3/kd_sp_T4/" --quiet

echo "════════════════ DONE (sp kd T=4) ════════════════"
date -u +%FT%TZ | aws s3 cp - "$S3/sp_kd_T4_complete"
