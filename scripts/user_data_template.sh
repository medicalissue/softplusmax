#!/usr/bin/env bash
# User-data executed at first boot by cloud-init.
# Placeholders replaced by render_user_data.sh:
#   @@VARIANT@@        sm | sp
#   @@WANDB_API_KEY@@
#   @@REPO_URL@@
#   @@REPO_REF@@
#   @@S3_PREFIX@@
#   @@AWS_REGION@@

set -euo pipefail
exec > /var/log/user-data.log 2>&1
echo "[user-data] $(date -u +%FT%TZ) starting on $(hostname)"

export AWS_DEFAULT_REGION="@@AWS_REGION@@"
export S3_PREFIX="@@S3_PREFIX@@"
export REPO_DIR=/workspace
export WANDB_API_KEY="@@WANDB_API_KEY@@"

# 1) Clone the repo (public).
mkdir -p /workspace
git clone --depth 1 --branch "@@REPO_REF@@" "@@REPO_URL@@" "$REPO_DIR"
cd "$REPO_DIR"

# 2) Reuse NELU's prebuilt venv (Python 3.10 + torch + timm + wandb).
#    Path /opt/nelu-venv/bin/activate is hard-coded into the venv at build time.
if [[ ! -x /opt/nelu-venv/bin/python3.10 ]]; then
    echo "[user-data] fetching prebuilt venv from S3"
    aws s3 cp s3://nelu-datasets/env/nelu-venv-py310-cu130.tar.gz /tmp/venv.tar.gz
    mkdir -p /opt
    tar xzf /tmp/venv.tar.gz -C /opt
    rm -f /tmp/venv.tar.gz
fi
ln -sf python3.10 /opt/nelu-venv/bin/python 2>/dev/null || true
# shellcheck disable=SC1091
source /opt/nelu-venv/bin/activate
python -V
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# 3) Top-up packages this experiment needs (small / fast).
pip install --quiet --upgrade tensorboard wandb || true

# 4) Hand off to the variant runner.
chmod +x scripts/run_*.sh
nohup bash scripts/run_@@VARIANT@@.sh > /var/log/runner.log 2>&1 || true
echo "[user-data] runner exited rc=$?"

# 5) Self-terminate.
echo "[user-data] self-terminating"
TOK=$(curl -sS -X PUT http://169.254.169.254/latest/api/token \
    -H "X-aws-ec2-metadata-token-ttl-seconds: 300" 2>/dev/null || true)
IID=$(curl -sS -H "X-aws-ec2-metadata-token: $TOK" \
    http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null || true)
if [[ -n "$IID" ]]; then
    aws ec2 terminate-instances --instance-ids "$IID" \
        --region "$AWS_DEFAULT_REGION" || shutdown -h +2 "self-terminate"
fi
