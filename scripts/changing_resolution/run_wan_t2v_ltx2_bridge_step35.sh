#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
lightx2v_path="${lightx2v_path:-$(cd "${script_dir}/../.." && pwd)}"
model_path="${model_path:-}"
ltx2_vae_ckpt="${ltx2_vae_ckpt:-}"
ltx2_upsampler_ckpt="${ltx2_upsampler_ckpt:-}"
seed="${seed:-42}"
prompt="${prompt:-Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.}"
negative_prompt="${negative_prompt:-camera shake, overexposed, blurry details, subtitles, low quality, worst quality, jpeg artifacts, deformed hands, deformed face, extra fingers, messy background}"
output_path="${output_path:-${lightx2v_path}/save_results/wan_t2v_ltx2_bridge_step35_seed${seed}.mp4}"
base_config="${lightx2v_path}/configs/changing_resolution/wan_t2v_ltx2_bridge_step35.json"
runtime_config="${lightx2v_path}/save_results/runtime_wan_t2v_ltx2_bridge_step35_seed${seed}.json"

if [[ -z "${model_path}" ]]; then
  echo "Please set model_path=/path/to/Wan2.1 model root" >&2
  exit 1
fi
if [[ -z "${ltx2_vae_ckpt}" || -z "${ltx2_upsampler_ckpt}" ]]; then
  echo "Please set ltx2_vae_ckpt and ltx2_upsampler_ckpt to local .safetensors files" >&2
  exit 1
fi

mkdir -p "$(dirname "${runtime_config}")"
python - "${base_config}" "${runtime_config}" "${ltx2_vae_ckpt}" "${ltx2_upsampler_ckpt}" <<'PY'
import json
import sys

base_config, runtime_config, vae_ckpt, upsampler_ckpt = sys.argv[1:5]
with open(base_config, "r", encoding="utf-8") as f:
    cfg = json.load(f)
cfg["ltx2_vae_ckpt"] = vae_ckpt
cfg["ltx2_upsampler_ckpt"] = upsampler_ckpt
with open(runtime_config, "w", encoding="utf-8") as f:
    json.dump(cfg, f, ensure_ascii=False, indent=4)
PY

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
source "${lightx2v_path}/scripts/base/base.sh"

python -m lightx2v.infer \
  --seed "${seed}" \
  --model_cls wan2.1_ltx2_bridge \
  --task t2v \
  --model_path "${model_path}" \
  --config_json "${runtime_config}" \
  --prompt "${prompt}" \
  --negative_prompt "${negative_prompt}" \
  --save_result_path "${output_path}"
