#!/bin/bash

# set path firstly
lightx2v_path=/data/yongyang/Jin/LightX2V
model_path=/data/yongyang/Jin/Wan-AI/Wan2.1-T2V-1.3B
wanupsampler_path=/data/yongyang/Jin/wanUpsampler

export CUDA_VISIBLE_DEVICES=0

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh
export PYTHONPATH=${wanupsampler_path}:${lightx2v_path}:$PYTHONPATH

prompts_file=${lightx2v_path}/scripts/changing_resolution/wan_t2v_batch20_prompts.txt
save_root=${lightx2v_path}/save_results/wan_t2v_wanupsampler_bridge_batch20
mkdir -p ${save_root}

negative_prompt="镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

index=1
while IFS= read -r prompt || [[ -n "$prompt" ]]; do
  if [[ -z "$prompt" ]]; then
    continue
  fi

  sample_id=$(printf "%02d" ${index})
  seed=$((41 + index))

  native_path=${save_root}/${sample_id}_native.mp4
  trilinear_path=${save_root}/${sample_id}_trilinear.mp4
  bridge_path=${save_root}/${sample_id}_wanupsampler.mp4
  compare_path=${save_root}/${sample_id}_compare.mp4

  python -m lightx2v.infer \
  --seed ${seed} \
  --model_cls wan2.1 \
  --task t2v \
  --model_path $model_path \
  --config_json ${lightx2v_path}/configs/changing_resolution/wan_t2v_native_512.json \
  --prompt "$prompt" \
  --negative_prompt "$negative_prompt" \
  --save_result_path ${native_path}

  python -m lightx2v.infer \
  --seed ${seed} \
  --model_cls wan2.1 \
  --task t2v \
  --model_path $model_path \
  --config_json ${lightx2v_path}/configs/changing_resolution/wan_t2v_512_05_step35.json \
  --prompt "$prompt" \
  --negative_prompt "$negative_prompt" \
  --save_result_path ${trilinear_path}

  python -m lightx2v.infer \
  --seed ${seed} \
  --model_cls wan2.1_wanupsampler_bridge \
  --task t2v \
  --model_path $model_path \
  --config_json ${lightx2v_path}/configs/changing_resolution/wan_t2v_wanupsampler_bridge_step35.json \
  --prompt "$prompt" \
  --negative_prompt "$negative_prompt" \
  --save_result_path ${bridge_path}

  ffmpeg -hide_banner -loglevel error -y \
    -i ${trilinear_path} \
    -i ${bridge_path} \
    -i ${native_path} \
    -filter_complex "hstack=inputs=3" \
    ${compare_path}

  index=$((index + 1))
done < ${prompts_file}
