#!/bin/bash

# set path and first
lightx2v_path=/data/yongyang/Jin/LightX2V
model_path=Lightricks/LTX-2

export CUDA_VISIBLE_DEVICES=0

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.infer \
--model_cls ltx2 \
--task t2av \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/ltx2/ltx2_upsample.json \
--prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
--negative_prompt "camera shake, overexposed, blurry details, subtitles, low quality, worst quality, jpeg artifacts, deformed hands, deformed face, extra fingers, messy background" \
--save_result_path ${lightx2v_path}/save_results/output_lightx2v_ltx2_t2av_compare_upsample.mp4
