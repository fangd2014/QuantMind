#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export QUANTMIND_ROTATION_REPORT_SCRIPT="/app/scripts/analysis/ths_concept_rotation_report.py"

exec "${SCRIPT_DIR}/run_concept_rotation_report.sh" "$@"
