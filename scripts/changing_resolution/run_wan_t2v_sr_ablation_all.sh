#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
lightx2v_path="${lightx2v_path:-$(cd "${script_dir}/../.." && pwd)}"
seed="${seed:-42}"
ablation_tag="${ablation_tag:-seed${seed}}"

common_env=(
  "lightx2v_path=${lightx2v_path}"
  "model_path=${model_path:-}"
  "seed=${seed}"
  "prompt=${prompt:-Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.}"
  "negative_prompt=${negative_prompt:-camera shake, overexposed, blurry details, subtitles, low quality, worst quality, jpeg artifacts, deformed hands, deformed face, extra fingers, messy background}"
)

run_case() {
  local name="$1"
  local script="$2"
  local output_path="${lightx2v_path}/save_results/${ablation_tag}_${name}.mp4"
  echo "==== Running ${name} ===="
  env "${common_env[@]}" \
    "ltx2_vae_ckpt=${ltx2_vae_ckpt:-}" \
    "ltx2_upsampler_ckpt=${ltx2_upsampler_ckpt:-}" \
    "output_path=${output_path}" \
    "${script_dir}/${script}"
}

run_case "A_trilinear_075_step25_512" "run_wan_t2v_cr_512_075_step25.sh"
run_case "B_trilinear_05_step25_512" "run_wan_t2v_cr_512_05_step25.sh"
run_case "C_bridge_05_step25_512" "run_wan_t2v_ltx2_bridge_step25.sh"
run_case "D_bridge_05_step35_512" "run_wan_t2v_ltx2_bridge_step35.sh"
run_case "E_bridge_05_step40_512" "run_wan_t2v_ltx2_bridge_step40.sh"
