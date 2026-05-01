#!/usr/bin/env bash
# Launch one g6.xlarge on-demand instance for the sm or sp pipeline.
# Reuses the same AMI / SG / subnet / IAM profile / SSH key as the
# AdaSoLU campaign so we don't re-provision AWS resources.
#
# Usage:
#   bash scripts/launch.sh sm        # softmax pipeline
#   bash scripts/launch.sh sp        # softplusmax pipeline
#
# Requires .env in repo root with AMI / SG / KEY / IAM_PROFILE /
# SUBNET_us_west_2{a..d} / WANDB_API_KEY / AWS_DEFAULT_REGION /
# REPO_URL / REPO_REF.
set -euo pipefail

VARIANT="${1:?variant required: sm | sp | sp_kd_T4 | sp_infonce | sqjr_infonce}"
case "$VARIANT" in
  sm|sp|sp_kd_T4|sp_infonce|sqjr_infonce) ;;
  *) echo "ERROR: invalid variant '$VARIANT'"; exit 2;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ENV_FILE:-$ROOT/.env}"
if [[ ! -f "$ENV_FILE" ]]; then
    echo "ERROR: $ENV_FILE not found. Copy .env.example → .env and fill it."; exit 2
fi
set -a; source "$ENV_FILE"; set +a

: "${AMI:?AMI missing}"
: "${SG:?SG missing}"
: "${KEY:?KEY missing}"
: "${IAM_PROFILE:?IAM_PROFILE missing}"
: "${WANDB_API_KEY:?WANDB_API_KEY missing}"
: "${REPO_URL:?REPO_URL missing}"
: "${REPO_REF:=main}"
: "${S3_PREFIX:=s3://nelu-checkpoints/softplusmax}"
: "${AWS_DEFAULT_REGION:=us-west-2}"
: "${INSTANCE_TYPE:=g6.xlarge}"
: "${AZ:=us-west-2a}"

subnet_var="SUBNET_$(echo "$AZ" | tr '-' '_')"
SUBNET="${!subnet_var:-}"
[[ -z "$SUBNET" ]] && { echo "ERROR: $subnet_var unset"; exit 2; }

# Render user-data with placeholder substitution.
USER_DATA=$(python3 - "$ROOT/scripts/user_data_template.sh" \
    "$VARIANT" "$WANDB_API_KEY" "$REPO_URL" "$REPO_REF" \
    "$S3_PREFIX" "$AWS_DEFAULT_REGION" <<'PY'
import sys, pathlib, base64
tpl, variant, wb, url, ref, s3, region = sys.argv[1:8]
text = pathlib.Path(tpl).read_text()
for k, v in {
    "@@VARIANT@@": variant, "@@WANDB_API_KEY@@": wb,
    "@@REPO_URL@@": url, "@@REPO_REF@@": ref,
    "@@S3_PREFIX@@": s3, "@@AWS_REGION@@": region,
}.items():
    text = text.replace(k, v)
sys.stdout.write(base64.b64encode(text.encode()).decode())
PY
)

NAME="softplusmax-${VARIANT}-$(date -u +%Y%m%dT%H%M%S)"
echo "▶ launching ${NAME} in ${AZ} (${INSTANCE_TYPE})"

aws ec2 run-instances \
    --region "$AWS_DEFAULT_REGION" \
    --image-id "$AMI" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY" \
    --subnet-id "$SUBNET" \
    --security-group-ids "$SG" \
    --iam-instance-profile "Name=$IAM_PROFILE" \
    --block-device-mappings '[
        {"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3","DeleteOnTermination":true}}
    ]' \
    --user-data "$USER_DATA" \
    --tag-specifications "ResourceType=instance,Tags=[
        {Key=Name,Value=$NAME},
        {Key=Project,Value=gate-norm},
        {Key=Role,Value=worker},
        {Key=Campaign,Value=softplusmax},
        {Key=Variant,Value=$VARIANT}
    ]" \
    --query 'Instances[0].[InstanceId,Placement.AvailabilityZone,State.Name]' \
    --output text
