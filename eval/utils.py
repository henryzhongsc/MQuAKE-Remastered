import random
from tqdm import tqdm
import torch
import json


def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings


def get_sent_embeddings(sents, contriever, tok, BSZ=32):
    all_embs = []
    for i in tqdm(range(0, len(sents), BSZ), disable=True):
        sent_batch = sents[i:i + BSZ]
        inputs = tok(sent_batch, padding=True, truncation=True, return_tensors='pt').to("cuda")
        with torch.no_grad():
            outputs = contriever(**inputs)
            embeddings = mean_pooling(outputs[0], inputs['attention_mask'])
        all_embs.append(embeddings.cpu())
    all_embs = torch.vstack(all_embs)
    return all_embs


def retrieve_facts(query, fact_embs, contriever, tok, k=1, threshold=float('-inf')):
    inputs = tok([query], padding=True, truncation=True, return_tensors='pt').to("cuda")
    with torch.no_grad():
        outputs = contriever(**inputs)
        query_emb = mean_pooling(outputs[0], inputs['attention_mask']).cpu()
    sim = (query_emb @ fact_embs.T)[0]
    knn = sim.topk(k, largest=True)

    if knn.values[0] < threshold:
        return None
    
    return knn.indices


def get_relation(subquestion, rels, rel_emb, contriever, tokenizer):
    rel_idx = retrieve_facts(subquestion, rel_emb, contriever, tokenizer, threshold=0.845)
    if rel_idx is None:
        return None
    rel = rels[rel_idx[0]]
    return rel


def call_model(prompt, stop, model, tokenzier, device='cuda', generate_length=50, temperature=1.0):
    encoding = tokenzier(prompt, return_tensors="pt")
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    gen_tokens = model.generate(
        input_ids,
        do_sample=True,
        attention_mask=attention_mask,
        max_length=len(input_ids[0]) + generate_length,
        stopping_criteria=stop,
        temperature=temperature,
        pad_token_id=tokenzier.eos_token_id
    )
    gen_text = tokenzier.batch_decode(gen_tokens)[0]
    gen_text = gen_text.replace(tokenzier.eos_token, '').replace(tokenzier.bos_token, '').strip()

    del input_ids, gen_tokens
    return gen_text
    
    
def call_model_template(prompt, stop, model, tokenzier, device, generate_length=50, temperature=1.0, front_space=4):
    template = '''A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {} ASSISTANT: \n'''
    input = template.format(prompt)
    encoding = tokenzier(input, return_tensors="pt")
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    gen_tokens = model.generate(
        input_ids,
        do_sample=True,
        attention_mask=attention_mask,
        max_length=len(input_ids[0]) + generate_length,
        stopping_criteria=stop,
        temperature=temperature,
        pad_token_id=tokenzier.eos_token_id
    )
    gen_text = tokenzier.batch_decode(gen_tokens)[0]
    gen_text = gen_text.replace(tokenzier.eos_token, '')
    
    del input_ids, gen_tokens

    return prompt + gen_text[front_space+len(input):]
    

def remove_extra_target_occurrences(gen, target, count):
    occurrences = gen.count(target)
    
    if occurrences <= count:
        return gen
    
    index = 0
    for _ in range(count + 1):
        index = gen.find(target, index) + len(target)
    
    return gen[:index - len(target) - 2]


def able_to_quit(gen, task_prompt):
    # The prompt contains 4 shots + 1 evaluation example -> 5.
    if gen.count('Final answer: ') >= 5:
        index = gen.find('Final answer: ', len(task_prompt)+13)
        ans = gen[index:]
        ans = ans.strip().split('\n')[0][len('Final answer: '):]
        if len(ans) > 0 and ans[-1] == '.':
            ans = ans[:-1]
        return True, ans
    else:
        return False, None


def get_rand_list(edit_num, seed, dataset_length):
    random.seed(seed)
    return random.sample(range(1, dataset_length + 1), edit_num)


def break_down_into_subquestions(d, subject, breakdown_prompt, sc_done, tokenizer, model, use_template=False, front_space=0):
    retval = []
    
    prompts = []
    for i in range(3):
        prompt = breakdown_prompt + f"Given this problem:\n{d['questions'][i]}\nExtract relations in square parentheses into follows:\n\"{subject}->"
        prompts.append(prompt)
    
    
    res = []
    for i in range(3):
        res.append(call_model(prompts[i], sc_done, model, tokenizer, temperature=0.2))
            
    
    for i in range(3):
        temp = res[i].split("\n\n")[4]
        temp = temp.strip().split("\n")[-1]
        rels = extract_entities(temp)
        retval.append(rels)
    return retval


def call_model_batch(prompts, stop, tokenzier, model, generate_length=50, temperature=1.0):
    # Tokenize the list of prompts
    input_ids = tokenzier(prompts, return_tensors="pt", padding=True, truncation=True).input_ids.cuda()
    
    # Generate text for the batch of prompts
    gen_tokens = model.generate(
        input_ids,
        do_sample=True,
        max_length=input_ids.shape[1] + generate_length,
        stopping_criteria=stop,
        temperature=temperature
    )
    
    # Decode the generated tokens to text for each prompt in the batch
    gen_texts = tokenzier.batch_decode(gen_tokens, skip_special_tokens=True)
    
    del input_ids, gen_tokens
    
    return gen_texts
    
    
def call_model_batch_template(prompts, stop, tokenzier, model, generate_length=150, temperature=1.0):
    template = '''A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {} ASSISTANT: \n'''
    input = [template.format(prompt) for prompt in prompts]
    # Tokenize the list of prompts
    input_ids = tokenzier(input, return_tensors="pt", padding=True, truncation=True).input_ids.cuda()

    # Generate text for the batch of prompts
    gen_tokens = model.generate(
        input_ids,
        do_sample=True,
        max_length=input_ids.shape[1]+generate_length,
        stopping_criteria=stop,
        temperature=temperature
    )

    # Decode the generated tokens to text for each prompt in the batch
    gen_texts = tokenzier.batch_decode(gen_tokens, skip_special_tokens=True)

    del input_ids, gen_tokens
    result = []
    for i, gen_text in enumerate(gen_texts):
      result.append(prompts[i] + gen_text[len(input[i]):])
    return result


def extract_entities(input_string):
    segments = input_string.split('->')
    
    # Initialize a list to hold the entities
    entities = []
    
    # Loop through each segment and check if it contains an entity within parentheses
    for segment in segments:
        if '(' in segment and ')' in segment:
            # Extract the entity by removing the parentheses
            entity = segment.strip()[1:-1]
            entities.append(entity)
    
    return entities


def get_ent_rel_id(file_path, dataset_name):
    if dataset_name in ["CF-3k", "CF-3k-old"]:
        dataset_name = "CF"
    if dataset_name in ["CF-3151", "CF-6334"]:
        dataset_name = "CF-9k"
    if dataset_name in ['T-old']:
        dataset_name = 'T'
    
    with open(f'{file_path}/datasets/{dataset_name}/entity2id.json', 'r') as f:
        entity2id = json.load(f)
    
    with open(f'{file_path}/datasets/{dataset_name}/id2entity.json', 'r') as f:
        id2entity = json.load(f)
    
    with open(f'{file_path}/datasets/{dataset_name}/rel2id.json', 'r') as f:
        rel2id = json.load(f)
    
    with open(f'{file_path}/datasets/{dataset_name}/id2rel.json', 'r') as f:
        id2rel = json.load(f)
    return entity2id, id2entity, rel2id, id2rel


def get_ent_alias(dataset):
    ent2alias = {}
    for idx, d in enumerate(dataset):
        answer = d['answer']
        answer_alias = d['answer_alias']
        ent2alias[answer] = set(answer_alias)

        answer = d['new_answer']
        answer_alias = d['new_answer_alias']
        ent2alias[answer] = set(answer_alias)

        for hop in d['single_hops']:
            answer = hop['answer']
            answer_alias = hop['answer_alias']
            ent2alias[answer] = set(answer_alias)

        for hop in d['new_single_hops']:
            answer = hop['answer']
            answer_alias = hop['answer_alias']
            ent2alias[answer] = set(answer_alias)
    
    return ent2alias


def get_fact_form_kg(subject, rel, kg_s_r_o, caseid, track):
    if subject in kg_s_r_o.keys():
        if rel in kg_s_r_o[subject].keys():
            fact_object, caseids = list(kg_s_r_o[subject][rel])

            # 1. if we want to mask, 
            # 2. if the case considered is not in the cases where the edit is from, 
            # 3. if the edit collided with one of the hop on the correct reasoning path.

            if True and caseid not in caseids and any(t == (subject, rel) for t in track):
                return "<no fact>", False, None
            
            fact = f'{rel.format(subject)} {fact_object}'
            return fact, True, fact_object
    
    return "<no fact>", False, None


def get_correct_track(d, edited, id2rel):
    tracks = []
    
    triples_name = 'new_triples' if edited else 'orig_triples'
    triples = d[triples_name]
    triples_labeled = d[triples_name + '_labeled']
    
    
    for i in range(len(triples)):
        tracks.append((triples_labeled[i][0], id2rel[triples[i][1]]))
    
    return tracks


def fit_subject_on_kg(subject, ent_emb, contriever, tokenizer, ents, kg_s_r_o, ent2alias):
    if subject in set(ents):
        return subject
    if len(ents) == 0:
        return subject
    indices = retrieve_facts(subject, ent_emb, contriever, tokenizer, k=min(10, len(ents)))
    for idx in indices:
        target = ents[idx]
        if target in ent2alias.keys() and subject in ent2alias[target]:
            return target
    return subject
    
    
def fetch_rel_subj2subq(subject, rel, relation2subq_prompt, sc_end_block, model, model_tokenizer, device):
    prompt = relation2subq_prompt + f"Given this relation: \"{rel}\" and this subject: \"{subject}\",\nThe corresponding question is"
    
    output = call_model(prompt, sc_end_block, model, model_tokenizer, temperature=0.2, generate_length=20, device=device)
    output = output.strip().split('\n\n')[5]
    output = output.strip().split('\n')[1]
    output = output.strip().split('\"')[1]
    
    return output
    
def extract_entity(questions, extract_entity_prompt, sc_end_block, model, model_tokenizer, device):
    for question in questions:
        task_prompt = extract_entity_prompt + f"Example 5:\nQuestion: \"{question}\"\nExtracted entity: ("
        output = call_model(task_prompt, sc_end_block, model, model_tokenizer, temperature=0.2, generate_length=20, device=device)
        
        output = output.strip().split('\n\n')[4]  
        output = output.strip().split('\n')[-1][len("Extracted entity: ("):]
        entity = output[:-1].strip()

        if entity in question:
            entity_start_idx = question.find(entity)
            entity_end_idx = entity_start_idx + len(entity) - 1

            while entity_end_idx + 1 < len(question):
                next_char = question[entity_end_idx + 1]

                # Continue if we find a period followed by a space OR an uppercase letter (allowing "W. G")
                if next_char == '.' and entity_end_idx + 2 < len(question):
                    next_space_idx = question.find(" ", entity_end_idx + 3)
                    if next_space_idx == -1:  # If no more spaces, take the rest of the string
                        entity_end_idx = len(question) - 1
                    else:
                        entity_end_idx = next_space_idx - 1
                else:
                    break

            full_entity = question[entity_start_idx: entity_end_idx+1].split("'")[0].replace('?', '')
            if full_entity in question:        
                return full_entity  

    return entity