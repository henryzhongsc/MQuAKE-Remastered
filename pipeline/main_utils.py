import logging
logger = logging.getLogger("main")

import argparse
import os
import json
import random

import torch
import numpy as np



def lock_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_desc', type=str, help='experiment description, this is purely cosmetic for readability purposes.')
    parser.add_argument('--pipeline_config_dir', type=str, help='file path of pipeline config.')
    parser.add_argument('--eval_config_dir', type=str, help='file path of eval config.')
    parser.add_argument('--output_folder_dir', default='', type=str, help='path of output model')
    parser.add_argument('--job_post_via', default='slurm_sbatch', type=str, help='slurm_sbatch or terminal')   

    
    parser.add_argument('--model_name', type=str, help='Model for the edits.')
    parser.add_argument('--device', type=str, help='Cuda or CPU?')
    parser.add_argument('--file_path', type=str, help='directory path to files')
    parser.add_argument('--seed', type=int, help='random seed number')
    parser.add_argument('--output_dir', type=str, help='output directory')
    parser.add_argument('--delete_duplicate_output_file', type=bool, help='Delete duplicate output file?')
    parser.add_argument('--edit_num', type=int, default=3000, help='number of questions to edit')
    parser.add_argument('--print_prompt', type=bool, default=False, help='print the prompt for debug')
    parser.add_argument('--dataset_name', type=str, default="CF-3k", help='default counterfactual')

    parser.add_argument('--postfix_breakdown_prompt', type=str, default='')
    parser.add_argument('--algo', type=str, default='mello')
    parser.add_argument('--masking', type=bool, default=True, help="whether to use masking")
    parser.add_argument('--start', type=int, default=0, help='start pos of dataset')
    parser.add_argument('--end', type=int, default=200, help='end pos of dataset')
    parser.add_argument('--use_template', type=bool, default=False, help='use template')
    parser.add_argument('--error_case', type=str, default="none", help='use template')
    args = parser.parse_args()

    if args.output_folder_dir != '':
        if args.output_folder_dir[-1] != '/':
            args.output_folder_dir  += '/'
    else:
        logger.error(f'Valid {args.output_folder_dir} is required.')

    return args


def register_args_and_configs(args):

    # Make outer output dir.
    if not os.path.isdir(args.output_folder_dir):
        os.makedirs(args.output_folder_dir)
        logger.info(f'Output folder dir {args.output_folder_dir} created.')
    else:
        logger.info(f'Output folder dir {args.output_folder_dir} already exist.')


    # Copy input eval config to output dir.
    with open(args.eval_config_dir) as eval_config_f:
        eval_config = json.load(eval_config_f)
        logger.info(f'Input eval config file {args.eval_config_dir} loaded.')
    
    
    # Make subdir under output dir to store input configs.
    input_config_subdir = eval_config['management']['sub_dir']['input_config']
    if not os.path.isdir(args.output_folder_dir + input_config_subdir):
        os.makedirs(args.output_folder_dir + input_config_subdir)
        logger.info(f'Input config subdir {args.output_folder_dir + input_config_subdir} created.')
    else:
        logger.info(f'Input config subdir {args.output_folder_dir + input_config_subdir} already exist.')

    input_eval_config_path = args.output_folder_dir + input_config_subdir + 'input_eval_config.json'
    with open(input_eval_config_path, "w+") as input_eval_config_f:
        json.dump(eval_config, input_eval_config_f, indent = 4)
        logger.info(f'Input eval config file {args.eval_config_dir} saved to {input_eval_config_path}.')

    # Copy input pipeline config to output dir.
    with open(args.pipeline_config_dir) as pipeline_config_f:
        pipeline_config = json.load(pipeline_config_f)
        logger.info(f'Input pipeline config file {args.pipeline_config_dir} loaded.')

    input_pipeline_config_path = args.output_folder_dir + input_config_subdir + 'input_pipeline_config.json'
    with open(input_pipeline_config_path, "w+") as input_pipeline_config_f:
        json.dump(pipeline_config, input_pipeline_config_f, indent = 4)
        logger.info(f'Input pipeline config file {args.pipeline_config_dir} saved to {input_pipeline_config_path}.')


    # Fuse and complete pipeline config, eval config, and args from argparser into a general config.
    config = dict()
    config['pipeline_params'] = pipeline_config['pipeline_params']
    config['eval_params'] = eval_config['eval_params']
    config['eval_results'] = dict() # processed result

    config['management'] = dict()
    config['management']['exp_desc'] = args.exp_desc
    config['management']['pipeline_config_dir'] = args.pipeline_config_dir
    config['management']['eval_config_dir'] = args.eval_config_dir
    config['management']['output_folder_dir'] = args.output_folder_dir
    config['management']['job_post_via'] = args.job_post_via
    if config['management']['job_post_via'] == 'slurm_sbatch':     # Add slurm info to config['management'] if the job is triggered via slurm sbatch.
        try:
            config['management']['slurm_info'] = register_slurm_sbatch_info()
        except Exception:
            config['management']['job_post_via'] == 'terminal'      # Likely not a slurm job, rollback to terminal post.
    config['management']['sub_dir'] = eval_config['management']['sub_dir']

    return config


def register_slurm_sbatch_info():
    slurm_job_id = os.environ['SLURM_JOB_ID']
    slurm_job_name = os.getenv('SLURM_JOB_NAME')
    slurm_out_file_dir = os.getenv('SLURM_SUBMIT_DIR') + '/slurm-' + os.getenv('SLURM_JOB_ID') + '.out'

    logger.info(f'Slurm job #{slurm_job_id} ({slurm_job_name}) running with slurm.out file at {slurm_out_file_dir}.')

    return {"slurm_job_id": slurm_job_id, "slurm_job_name": slurm_job_name, "slurm_out_file_dir": slurm_out_file_dir}



def register_result(processed_results, raw_results, config):
    with open(os.path.join(config['management']['output_folder_dir'], "raw_answer_dict.json"), 'w') as f:
        json.dump(raw_results, f)
    with open(os.path.join(config['management']['output_folder_dir'], "results.json"), 'w') as f:
        json.dump({"result": processed_results}, f)


    config['eval_results']['processed_results'] = processed_results
    logger.info('Experiments concluded, below is the raw_results: ')
    logger.info(json.dumps(raw_results, indent=4))

    logger.info('##### And below is the processed_results: #####')
    logger.info(json.dumps(config['eval_results']['processed_results'], indent=4))


def register_exp_time(start_time, end_time, config):
    config['management']['start_time'] = str(start_time)
    config['management']['end_time'] = str(end_time)
    config['management']['exp_duration'] = str(end_time - start_time)


def register_output_config(config):
    output_config_path = config['management']['output_folder_dir'] + config['management']['sub_dir']['output_config']
    with open(output_config_path, "w+") as output_config_f:
        json.dump(config, output_config_f, indent = 4)
        logger.info(f'output_config file saved to {output_config_path}.')