import logging
logger = logging.getLogger("main")

import torch

import eval.mquake_remastered.main as mquake_main
import pipeline.inference as inference
import pipeline.inference_mquake as inference_mquake
import pipeline.gwalk.eval_loop as eval_loop


def eval_gwalk(config):
    eval_config = config['configs']['eval_config']
    pipeline_config = config['configs']['pipeline_config']

    # Load the MQuAKE-Remastered dataset
    mquake_dataset = mquake_main.prepare_mquake_input(config)

    # Initialize the main LLM using the template's inference module (device_map="auto")
    model, tokenizer = inference.initialize_model_tokenizer(pipeline_config)

    # Initialize Contriever on explicit device (small model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    contriever, contriever_tokenizer = inference_mquake.initialize_contriever(device=str(device))

    # Get stopping criteria for the model
    model_name = pipeline_config['model_name']
    if model_name not in inference_mquake.MODEL_to_SC:
        raise ValueError(f"Model {model_name} not found in MODEL_to_SC. "
                         f"Supported models: {list(inference_mquake.MODEL_to_SC.keys())}")
    stopping_criteria_dict = inference_mquake.MODEL_to_SC[model_name]

    # Build eval_var dict expected by eval loop
    eval_var = {
        'model': model,
        'tokenizer': tokenizer,
        'contriever': contriever,
        'contriever_tokenizer': contriever_tokenizer,
        'stopping_criteria_dict': stopping_criteria_dict,
        'mquake_remastered_dataset': mquake_dataset,
        'device': device
    }

    result_summary_str, raw_answer_dict = eval_loop.gwalk_eval_loop(eval_var, config)

    # Convert to template's (raw_results, processed_results) format
    processed_results = {
        "total": mquake_dataset.get_length(),
        "metrics": {
            "mquake_accuracy": result_summary_str
        }
    }

    raw_results = {
        "summary": processed_results,
        "details": raw_answer_dict
    }

    logger.info(f"G-Walk evaluation complete: {result_summary_str}")

    return raw_results, processed_results
