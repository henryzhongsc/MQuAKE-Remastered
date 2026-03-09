import os
import logging
logger = logging.getLogger("main")

from datasets import Dataset as HFDataset

import eval.mquake_remastered.mquake_utils as utils
from pipeline.inference_mquake import REL2SUBQ as rel2subq


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


def gwalk_eval_loop(eval_var, config):
    prompts_dir = config['configs']['eval_config']['prompts_dir']

    with open(os.path.join(prompts_dir, 'fill_out_ga_w_blank2.txt'), 'r', encoding='utf-8') as f:
        task_prompt = f.read()

    with open(os.path.join(prompts_dir, 'subq_breakdown.txt'), 'r', encoding='utf-8') as f:
        breakdown_prompt = f.read()

    with open(os.path.join(prompts_dir, 'relation2subq_prompt2.txt'), 'r', encoding='utf-8') as f:
        relation2subq_prompt = f.read()

    with open(os.path.join(prompts_dir, 'extract_entity.txt'), 'r', encoding='utf-8') as f:
        extract_entity_prompt = f.read()

    mr_dataset = eval_var['mquake_remastered_dataset']
    rand_list = mr_dataset.get_randlist()
    raw_answer_dict = {}

    kg_s_r_o, rels, ents, id2rel = mr_dataset.process_kg()
    rel_emb = utils.get_sent_embeddings(rels, eval_var['contriever'], eval_var['contriever_tokenizer'])
    ent_emb = utils.get_sent_embeddings(ents, eval_var['contriever'], eval_var['contriever_tokenizer'])

    ent2alias = utils.get_ent_alias(mr_dataset.get_dataset())

    dataset = _limit_dataset(mr_dataset.get_dataset(), config['configs']['eval_config'].get('max_eval_instances', None))
    for d in dataset:
        edit_flag = d['case_id'] in rand_list
        raw_answer_dict[d['case_id']] = {'edited': edit_flag}

        start_subject = utils.extract_entity(d["questions"], extract_entity_prompt, eval_var['stopping_criteria_dict']['done'], eval_var['model'], eval_var['tokenizer'], device=eval_var['device'])
        breakdown_rels_list = utils.break_down_into_subquestions(d, start_subject, breakdown_prompt, eval_var['stopping_criteria_dict']['done'], eval_var['tokenizer'], eval_var['model'])
        llm_answers = []
        for q_id, q in enumerate(d["questions"]):
            subject = str(start_subject)
            breakdown_rels = breakdown_rels_list[q_id]

            ans = None
            prompt = task_prompt + "\n\nQuestion: " + q + "\n"
            for i in range(len(breakdown_rels)):
                subject = utils.fit_subject_on_kg(subject, ent_emb, eval_var['contriever'], eval_var['contriever_tokenizer'], ents, kg_s_r_o, ent2alias)

                relation = breakdown_rels[i]
                rel = utils.get_relation(relation, rels, rel_emb, eval_var['contriever'], eval_var['contriever_tokenizer'])
                if rel is not None:
                    subquestion = rel2subq.get(rel, rel).format(subject)
                else:
                    subquestion = utils.fetch_rel_subj2subq(subject, relation, relation2subq_prompt,
                                                            eval_var['stopping_criteria_dict']['end_block'],
                                                            model=eval_var['model'],
                                                            model_tokenizer=eval_var['tokenizer'], device=eval_var['device'])
                prompt = prompt + "Subquestion: " + subquestion + "\n"

                prompt = utils.call_model(prompt, eval_var['stopping_criteria_dict']['facts'], eval_var['model'], eval_var['tokenizer'])

                if prompt.strip().split('\n')[-1] == 'Retrieved fact:':
                    prompt = prompt[:-len('\nRetrieved fact:')]
                prompt = utils.remove_extra_target_occurrences(prompt, "Question: ", 5)

                temp_split = prompt.strip().split('\n')
                if len(temp_split) < 2:
                    break  # failed case

                generated_answer = temp_split[-1][len("Generated answer: "):]

                ga_seg = generated_answer.strip().split('. ')

                if len(ga_seg) >= 2:
                    answer_object = ". ".join(ga_seg[1:])
                else:
                    break
                fact_sent, contra_or_not, fact_object = utils.get_fact_form_kg(subject, rel, kg_s_r_o, d['case_id'],
                                                                               utils.get_correct_track(d, edit_flag, id2rel))

                contra_promt = "Retrieved fact {} to generated answer, so continue with this subject: {}.\n"
                if contra_or_not:
                    does_or_doesnot = "contradicts"
                    inter_answer = fact_object
                else:
                    does_or_doesnot = "does not contradict"
                    inter_answer = answer_object

                contra_promt = contra_promt.format(does_or_doesnot, inter_answer)

                subject = utils.fit_subject_on_kg(inter_answer, ent_emb, eval_var['contriever'], eval_var['contriever_tokenizer'], ents, kg_s_r_o, ent2alias)

                ans = subject
                prompt = prompt + '\nRetrieved fact: ' + fact_sent + '.\n' + contra_promt

            llm_answers.append(ans)

            if mr_dataset.check_answer(edit_flag, d, ans, q_id):
                break

        raw_answer_dict[d['case_id']]['answers'] = llm_answers
        logger.info(mr_dataset.get_result_summary())

    return mr_dataset.get_result_summary(), raw_answer_dict
