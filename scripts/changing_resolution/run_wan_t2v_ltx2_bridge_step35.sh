#!/bin/bash

# set path firstly
lightx2v_path=/data/yongyang/Jin/LightX2V
model_path=/data/yongyang/Jin/Wan-AI/Wan2.1-T2V-1.3B

export CUDA_VISIBLE_DEVICES=0

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

# Before running, set local ltx2_vae_ckpt and ltx2_upsampler_ckpt in:
# ${lightx2v_path}/configs/changing_resolution/wan_t2v_ltx2_bridge_step35.json
python -m lightx2v.infer \
--seed 42 \
--model_cls wan2.1_ltx2_bridge \
--task t2v \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/changing_resolution/wan_t2v_ltx2_bridge_step35.json \
--prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
--negative_prompt "camera shake, overexposed, blurry details, subtitles, low quality, worst quality, jpeg artifacts, deformed hands, deformed face, extra fingers, messy background" \
--save_result_path ${lightx2v_path}/save_results/wan_t2v_ltx2_bridge_step35_seed42.mp4
