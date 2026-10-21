#!/bin/bash

profile_dir="python/sglang/srt/dynamic_sp/work/profile-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$profile_dir"

nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  --wait=all \
  --output="$profile_dir/dynamic-sp" \
  bash ./launch_dynamic_sp_8gpu.sh \
  --dtype bfloat16 --no-verify --offered-load 25000 --scheduler greedy
