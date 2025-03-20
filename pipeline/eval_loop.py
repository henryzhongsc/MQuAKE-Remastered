import os
import logging
logger = logging.getLogger("main")
import eval.utils as utils

from inference import REL2SUBQ as rel2subq


def mello_eval_loop(eval_var, config):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(current_dir, '../'))

    with open(os.path.join(base_dir, 'prompts', 'MeLLo-prompt.txt'), 'r', encoding='utf-8') as f:
        task_prompt = f.read()
        
    tot = 0
    mr_dataset = eval_var['mquake_remastered_dataset']
    rand_list = mr_dataset.get_randlist()
    raw_answer_dict = {}
    for d in mr_dataset.get_dataset():
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

    


def gwalk_eval_loop(eval_var, config):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(current_dir, '../'))

    with open(os.path.join(base_dir, 'prompts', 'fill_out_ga_w_blank2.txt'), 'r', encoding='utf-8') as f:
        task_prompt = f.read()

    with open(os.path.join(base_dir, 'prompts', 'subq_breakdown.txt'), 'r', encoding='utf-8') as f:
        breakdown_prompt = f.read()

    with open(os.path.join(base_dir, 'prompts', 'relation2subq_prompt2.txt'), 'r', encoding='utf-8') as f:
        relation2subq_prompt = f.read()
    
    with open(os.path.join(base_dir, 'prompts', 'extract_entity.txt'), 'r', encoding='utf-8') as f:
        extract_entity_prompt = f.read()
            
    mr_dataset = eval_var['mquake_remastered_dataset']
    rand_list = mr_dataset.get_randlist()
    raw_answer_dict = {}

    kg_s_r_o, rels, ents, id2rel = mr_dataset.process_kg()
    rel_emb = utils.get_sent_embeddings(rels, eval_var['contriever'], eval_var['contriever_tokenizer'])
    ent_emb = utils.get_sent_embeddings(ents, eval_var['contriever'], eval_var['contriever_tokenizer'])

    ent2alias = utils.get_ent_alias(mr_dataset.get_dataset())

    for d in mr_dataset.get_dataset():
        edit_flag = d['case_id'] in rand_list
        raw_answer_dict[d['case_id']] = {'edited': edit_flag}

        start_subject = utils.extract_entity(d["questions"], extract_entity_prompt, eval_var['stopping_criteria_dict']['done'], eval_var['model'], eval_var['tokenizer'], device=eval_var['device'])
        breakdown_rels_list = utils.break_down_into_subquestions(d, start_subject, breakdown_prompt, eval_var['stopping_criteria_dict']['done'], eval_var['tokenizer'], eval_var['model'])
        # logger.info(f"POTENTIAL ENTITY: {entity_potential}")
        llm_answers = []
        for q_id, q in enumerate(d["questions"]):
            # logger.info(f"==============================||q_id = {q_id}||==============================")
            # logger.info(f"QUESTION: {q}")
            subject = str(start_subject)
            breakdown_rels = breakdown_rels_list[q_id]
            # logger.info(len(breakdown_rels))
            
            ans = None
            prompt = task_prompt + "\n\nQuestion: " + q + "\n"
            for i in range(len(breakdown_rels)):
                subject = utils.fit_subject_on_kg(subject, ent_emb, eval_var['contriever'], eval_var['contriever_tokenizer'], ents, kg_s_r_o, ent2alias)
                
                # logger.info(f"SUBJECT: {subject}")
                relation = breakdown_rels[i]
                rel = utils.get_relation(relation, rels, rel_emb, eval_var['contriever'], eval_var['contriever_tokenizer'])
                if rel is not None:
                    subquestion = rel2subq.get(rel, rel).format(subject)
                    # logger.info(f"relation: {relation} || {rel}")
                else:
                    # logger.info(f"relation: {relation} || ")
                    subquestion = utils.fetch_rel_subj2subq(subject, relation, relation2subq_prompt,
                                                      eval_var['stopping_criteria_dict']['end_block'],
                                                      model=eval_var['model'],
                                                      model_tokenizer=eval_var['tokenizer'], device=eval_var['device'])
                # logger.info(f"SubQ: {subquestion}")  
                prompt = prompt + "Subquestion: " + subquestion + "\n"
                
                prompt = utils.call_model(prompt, eval_var['stopping_criteria_dict']['facts'], eval_var['model'], eval_var['tokenizer'])

                if prompt.strip().split('\n')[-1] == 'Retrieved fact:':
                    prompt = prompt[:-len('\nRetrieved fact:')]
                prompt = utils.remove_extra_target_occurrences(prompt, "Question: ", 5)
                
                temp_split = prompt.strip().split('\n')
                # otherwise, extract the generated subquestion
                if len(temp_split) < 2:
                    break  # failed case
                
                generated_answer = temp_split[-1][len("Generated answer: "):]
                
                # Genertaed answer: XX is {}. YY
                ga_seg = generated_answer.strip().split('. ')
                
                if len(ga_seg) >= 2:
                    answer_object = ". ".join(ga_seg[1:])
                else:
                    break
                # logger.info(f"TEMP ANSWER: {answer_object}")
                fact_sent, contra_or_not, fact_object = utils.get_fact_form_kg(subject, rel, kg_s_r_o, d['case_id'],
                                                                         utils.get_correct_track(d, edit_flag, id2rel))
                
                # check whether there is a contradiction:
                # contra_promt = "Retrieved fact {} to generated answer, so the intermediate answer is: {}\n"
                contra_promt = "Retrieved fact {} to generated answer, so continue with this subject: {}.\n"
                if contra_or_not:
                    does_or_doesnot = "contradicts"
                    inter_answer = fact_object
                else:
                    does_or_doesnot = "does not contradict"
                    inter_answer = answer_object
                
                contra_promt = contra_promt.format(does_or_doesnot, inter_answer)
                
                # reset pointer and var for the next hop:
                subject = utils.fit_subject_on_kg(inter_answer, ent_emb, eval_var['contriever'], eval_var['contriever_tokenizer'], ents, kg_s_r_o, ent2alias)
                # subject = inter_answer
                
                
                ans = subject
                prompt = prompt + '\nRetrieved fact: ' + fact_sent + '.\n' + contra_promt
                
                # logger.info(f"{str(contra_or_not)}, {fact_sent}")
                # logger.info(f"Hop answer: {ans}")
                # logger.info("------------------------------------------")
               

            llm_answers.append(ans)
            # logger.info(prompt[len(task_prompt):])
            # logger.info(ans)

            if mr_dataset.check_answer(edit_flag, d, ans, q_id):
                break
            
        raw_answer_dict[d['case_id']]['answers'] = llm_answers
        logger.info(mr_dataset.get_result_summary())

    return mr_dataset.get_result_summary(), raw_answer_dict


EVAL_LOOPS = {
    'mello': mello_eval_loop,
    'gwalk': gwalk_eval_loop
}