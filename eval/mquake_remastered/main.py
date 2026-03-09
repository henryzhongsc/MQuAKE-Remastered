import logging
logger = logging.getLogger("main")

from eval.mquake_remastered.mquake_dataset import MQuAKE_Remastered


def prepare_mquake_input(config):
    """Load and prepare the MQuAKE-Remastered dataset for evaluation."""
    eval_config = config['configs']['eval_config']

    dataset_name = eval_config['dataset_name']
    edit_num = eval_config['edit_num']

    mquake_dataset = MQuAKE_Remastered(dataset_name, edit_num)

    logger.info(f'MQuAKE-Remastered dataset loaded: {dataset_name} with {edit_num} edits, '
                f'{mquake_dataset.get_length()} total cases, {len(mquake_dataset.get_randlist())} edited cases.')

    return mquake_dataset
