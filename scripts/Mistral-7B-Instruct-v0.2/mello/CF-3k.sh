#!/bin/bash

output_folder_root_dir=$1

# Fall back to default_output_dir from global_setting.py if not provided
if [ -z "$output_folder_root_dir" ]; then
    output_folder_root_dir=$(python -c "from configs.global_setting import default_output_dir; print(default_output_dir)")
fi

uv run python pipeline/main.py \
    --exp_desc Mistral-7B-Instruct-v0.2__mello__CF-3k_1000 \
    --pipeline_config_dir configs/pipeline_config/Mistral-7B-Instruct-v0.2/mello.json \
    --eval_config_dir configs/eval_config/mquake_remastered/CF-3k_1000.json \
    --management_config_dir configs/management_config/default.json \
    --output_folder_dir ${output_folder_root_dir}/mquake_remastered/Mistral-7B-Instruct-v0.2/mello/CF-3k_1000/
