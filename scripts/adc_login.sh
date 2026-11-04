#!/usr/bin/env bash
# Interactive (opens a browser): log in Application Default Credentials with the scopes SpriteSheet needs,
# then set the quota project. Usage: bash scripts/adc_login.sh <GCP_PROJECT_ID>
set -euo pipefail
PROJECT="${1:?usage: bash scripts/adc_login.sh <GCP_PROJECT_ID>}"
S=https://www.googleapis.com/auth
gcloud auth application-default login --scopes="$S/spreadsheets,$S/drive.file,$S/cloud-platform"
gcloud auth application-default set-quota-project "$PROJECT"
echo "ADC ready. Now run: python3 -I scripts/sheets_probe.py"
