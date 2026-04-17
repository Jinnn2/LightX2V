#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
lightx2v_path="${lightx2v_path:-$(cd "${script_dir}/../.." && pwd)}"
model_path="${model_path:-}"
seed="${seed:-42}"
prompt="${prompt:-Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.}"
negative_prompt="${negative_prompt:-camera shake, overexposed, blurry details, subtitles, low quality, worst quality, jpeg artifacts, deformed hands, deformed face, extra fingers, messy background}"
output_path="${output_path:-${lightx2v_path}/save_results/wan_t2v_cr_512_075_step25_seed${seed}.mp4}"

if [[ -z "${model_path}" ]]; then
  echo "Please set model_path=/path/to/Wan2.1 model root" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
source "${lightx2v_path}/scripts/base/base.sh"

python -m lightx2v.infer \
  --seed "${seed}" \
  --model_cls wan2.1 \
  --task t2v \
  --model_path "${model_path}" \
  --config_json "${lightx2v_path}/configs/changing_resolution/wan_t2v_512_075_step25.json" \
  --prompt "${prompt}" \
  --negative_prompt "${negative_prompt}" \
  --save_result_path "${output_path}"
