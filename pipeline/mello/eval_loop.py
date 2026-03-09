import os
import logging
logger = logging.getLogger("main")

from datasets import Dataset as HFDataset

import eval.mquake_remastered.mquake_utils as utils


def _limit_dataset(dataset, max_eval):
    """Limit dataset to max_eval instances, handling both HF Dataset and list types."""
    if max_eval is None:
        return dataset
    if isinstance(dataset, HFDataset):
        dataset = dataset.select(range(min(max_eval, len(dataset))))
    else:
        dataset = dataset[:max_eval]
    logger.info(f"Limiting evaluation to {max_eval} instances.")
    return dataset


def mello_eval_loop(eval_var, config):
    prompts_dir = config['configs']['eval_config']['prompts_dir']
    with open(os.path.join(prompts_dir, 'MeLLo-prompt.txt'), 'r', encoding='utf-8') as f:
        task_prompt = f.read()

    tot = 0
    mr_dataset = eval_var['mquake_remastered_dataset']
    rand_list = mr_dataset.get_randlist()
    raw_answer_dict = {}
    dataset = _limit_dataset(mr_dataset.get_dataset(), config['configs']['eval_config'].get('max_eval_instances', None))
    for d in dataset:
        edit_flag = d['case_id'] in rand_list
        raw_answer_dict[d['case_id']] = {'edited': edit_flag}
        if rand_list:
            new_facts, _, _, _ = mr_dataset.get_edits_without_contamination(d)
            if not new_facts:
                new_facts = ["No relevant fact."]
            embs = utils.get_sent_embeddings(new_facts, eval_var['contriever'], eval_var['contriever_tokenizer'])

        tot += 1
        llm_answers = []

        for qid, q in enumerate(d["questions"]):
            prompt = task_prompt + "\n\nQuestion: " + q
            ans = None

            for i in range(4):  # max of 4 hops
                # prompt the model to generate a subquestion and a tentative answer
                prompt = utils.call_model(prompt, eval_var['stopping_criteria_dict']['facts'], eval_var['model'], eval_var['tokenizer'])
                if prompt.strip().split('\n')[-1] == 'Retrieved fact:':
                    prompt = prompt[:-len('\nRetrieved fact:')]
                prompt = utils.remove_extra_target_occurrences(prompt, "Question: ", 5)

                # if final answer is there, get the answer and exit
                quit, ans = utils.able_to_quit(prompt, task_prompt)
                if quit:
                    break

                temp_split = prompt.strip().split('\n')
                # otherwise, extract the generated subquestion
                if len(temp_split) < 2:
                    break  # failed case

                subquestion = temp_split[-2]

                if not subquestion.startswith('Subquestion: '):
                    break  # failed case

                if rand_list:
                    fact_ids = utils.retrieve_facts(subquestion, embs, eval_var['contriever'], eval_var['contriever_tokenizer'])
                    fact_sent = new_facts[fact_ids[0]]

                    # put the retrieved fact at the end of the prompt, the model self-checks if it contradicts
                    prompt = prompt + '\nRetrieved fact: ' + fact_sent + "\n"

                quit, ans = utils.able_to_quit(prompt, task_prompt)
                if quit:
                    break

            llm_answers.append(ans)

            if mr_dataset.check_answer(edit_flag, d, ans, qid):
                break

        raw_answer_dict[d['case_id']]['answers'] = llm_answers
        logger.info(mr_dataset.get_result_summary())

    return mr_dataset.get_result_summary(), raw_answer_dict
