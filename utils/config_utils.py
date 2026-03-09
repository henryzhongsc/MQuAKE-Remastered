import logging
logger = logging.getLogger("main")

import os
import json

def register_raw_and_processed_results(raw_results, processed_results, config):

    management_config = config['configs']['management_config']

    if raw_results is not None:
        raw_results_path = os.path.join(management_config['output_folder_dir'], management_config['sub_dir']['raw_results_folder'], management_config['sub_dir']['raw_results_file'])
        with open(raw_results_path, "w+") as raw_results_f:
            json.dump(raw_results, raw_results_f, indent = 4)
            logger.info(f'raw_results file saved to {raw_results_path}.')
    else:
        logger.info(f'raw_results is {raw_results}')


    if processed_results is not None:
        config['processed_results'] = processed_results
    else:
        logger.error(f'processed_results is {processed_results}.')


    logger.info('Experiments concluded, showing raw_results below: ')
    logger.info(json.dumps(raw_results, indent=4))

    logger.info('##### Showing processed_results below #####')
    logger.info(json.dumps(config['processed_results'], indent=4))


def register_exp_time(start_time, end_time, management_config):
    management_config['start_time'] = str(start_time)
    management_config['end_time'] = str(end_time)
    management_config['exp_duration'] = str(end_time - start_time)


def register_output_file(config):
    output_config_path = os.path.join(config['configs']['management_config']['output_folder_dir'], config['configs']['management_config']['sub_dir']['output_file'])
    with open(output_config_path, "w+") as output_f:
        json.dump(config, output_f, indent = 4)
        logger.info(f'Output file saved to {output_config_path}.')