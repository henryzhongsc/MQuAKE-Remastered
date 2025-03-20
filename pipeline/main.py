import json
import logging

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(current_dir, '../'))
sys.path.append(base_dir)
os.chdir(base_dir)
print(sys.path)
from config.access_tokens import hf_access_token
from huggingface_hub import login
login(token=hf_access_token)

import pipeline.main_utils as main_utils

from eval_mquake_remastered import eval_mquake_remastered
SEED = 100
main_utils.lock_seed(SEED)

args = main_utils.parse_args()
config = main_utils.register_args_and_configs(args)
logger = logging.getLogger("main")
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)


logger.info(f"Experiment {config['management']['exp_desc']} (SEED={SEED}) started with the following config: ")
logger.info(json.dumps(config, indent=4))

processed_results, raw_results = eval_mquake_remastered(config)
main_utils.register_result(processed_results, raw_results, config)