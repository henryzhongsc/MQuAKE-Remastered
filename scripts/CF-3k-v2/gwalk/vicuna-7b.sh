output_dir_root=$1

dataset="CF-3k-v2"
model="vicuna-7b-v1.5"
method="gwalk"
editnum=1000

python pipeline/main.py \
--exp_desc ${dataset}_${model}_${method}_${editnum} \
--pipeline_config_dir ${output_dir_root}/config/pipeline_config/${method}/${model}.json \
--eval_config_dir ${output_dir_root}/config/eval_config/${dataset}/${editnum}.json \
--output_folder_dir ${output_dir_root}/outputs/${method}/${model}/${dataset}/${editnum}