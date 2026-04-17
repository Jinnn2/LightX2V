#!/bin/bash
set -euo pipefail

# Backward-compatible default bridge entry. This runs the step25 variant.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${script_dir}/run_wan_t2v_ltx2_bridge_step25.sh" "$@"
