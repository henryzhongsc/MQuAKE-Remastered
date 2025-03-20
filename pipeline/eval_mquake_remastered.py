import logging
logger = logging.getLogger("main")

import torch
import inference as inference
import pipeline.eval_loop as eval_loop
from eval.mquake import MQuAKE_Remastered



def eval_mquake_remastered(config):
    eval_params = config['eval_params']
    pipeline_params = config['pipeline_params']

    mquake_remastered_dataset = MQuAKE_Remastered(eval_params['dataset_name'], eval_params['edit_num'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, tokenizer, contriever, contriever_tokenizer, stopping_criteria_dict = inference.initialize_model_tokenizer(pipeline_params, device)
    eval_var = {'model': model, 
        'tokenizer': tokenizer, 
        'contriever': contriever, 
        'contriever_tokenizer': contriever_tokenizer, 
        'stopping_criteria_dict': stopping_criteria_dict,
        'mquake_remastered_dataset': mquake_remastered_dataset,
        'device': device
    }

    return eval_loop.EVAL_LOOPS[pipeline_params['method']](eval_var, config)
    