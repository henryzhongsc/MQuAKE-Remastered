from torch.utils.data import Dataset
from datasets import load_dataset


class MQuAKE_Remastered(Dataset):
    def __init__(self, dataset_name, edit_num):
        if dataset_name in ['CF-3k', 'CF-9k', 'CF-6334', 'T']:
            self.dataset_name = dataset_name
        else:
            raise ValueError("Dataset name <%s> unknown." % dataset_name)

        self.dataset = load_dataset("henryzhongsc/MQuAKE-Remastered")[dataset_name.replace('-', '')]

        self.question_type_dict = {}
        self.type_correctness = {}
        self.rand_list = []
        if dataset_name == 'CF-6334':
            self.type_correctness = {
                "train_edited": [0, 0],
                "test_edited": [0, 0],
                "unedited": [0, 0]
            }
            processed_dataset = []

            for d in self.dataset:
                labels = d['split'][str(edit_num)]

                # train edited:
                if 'train_edited' in labels:
                    processed_dataset.append(d)
                    self.rand_list.append(d['case_id'])
                    self.question_type_dict[d['case_id']] = 'train_edited'

                # test edited unique:
                if 'test_edited' in labels and not 'train_edited' in labels:
                    processed_dataset.append(d)
                    self.rand_list.append(d['case_id'])
                    self.question_type_dict[d['case_id']] = 'test_edited'

                if 'test_unedited' in labels:
                    processed_dataset.append(d)
                    self.question_type_dict[d['case_id']] = 'unedited'

            self.dataset = processed_dataset
        else:
            self.type_correctness = {
                "edited": [0, 0],
                "unedited": [0, 0]
            }

            for d in self.dataset:
                labels = d['split'][str(edit_num)]
                if 'edited' in labels:
                    self.rand_list.append(d['case_id'])
                    self.question_type_dict[d['case_id']] = 'edited'
                else:
                    self.question_type_dict[d['case_id']] = 'unedited'

        self.length = len(self.dataset)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise IndexError("Index out of range.")
        return self.dataset[idx]

    def get_length(self):
        return self.length

    def get_dataset(self):
        return self.dataset

    def get_randlist(self):
        return self.rand_list

    def process_kg(self):
        kg_s_r_o = {}

        rels = set()
        ents = set()

        id2rel = {}
        rel2id = {}

        for d in self.dataset:
            if d['case_id'] not in self.rand_list:
                continue
            caseid = d['case_id']
            for edit in d["requested_rewrite"]:
                s = edit['subject']
                o = edit['target_new_str']
                o_pre = edit['target_true_str']

                rel_id = edit['relation_id']
                r = edit['prompt']

                if r not in rel2id.keys():
                    rel2id[r] = rel_id
                    id2rel[rel_id] = r
                else:
                    assert rel2id[r] == rel_id

                rels.add(r)
                ents.add(s)
                ents.add(o)
                ents.add(o_pre)

                if s in kg_s_r_o.keys():
                    if r in kg_s_r_o[s].keys():
                        obj = kg_s_r_o[s][r][0]
                        id_set = kg_s_r_o[s][r][1]
                        id_set.add(caseid)
                        kg_s_r_o[s][r] = [obj, id_set]
                    else:
                        kg_s_r_o[s][r] = [o, set([caseid])]
                else:
                    kg_s_r_o[s] = {r: [o, set([caseid])]}

        return kg_s_r_o, list(rels), list(ents), id2rel

    def get_result_summary(self):
        return " | ".join([f"{key}: {value[0]}, {value[1]}" for key, value in self.type_correctness.items()])

    def check_answer(self, edit_flag, instance, ans, qid):
        if ans is None:
            ans = "no answer is provided. This is a failed case."
        # Define answer and answer_alias keys based on edit_flag
        answer = "answer"
        answer_alias = "answer_alias"
        if edit_flag:
            answer = "new_" + answer
            answer_alias = "new_" + answer_alias

        # Convert the answer and ans to upper case
        ans_upper = ans.upper()
        instance_answer_upper = instance[answer].upper()

        # Convert each alias to upper case for comparison
        instance_answer_alias_upper = [alias.upper() for alias in instance[answer_alias]]

        if ans_upper == instance_answer_upper or ans_upper in instance_answer_alias_upper:
            self.type_correctness[self.question_type_dict[instance['case_id']]][0] += 1
            self.type_correctness[self.question_type_dict[instance['case_id']]][1] += 1
            return True

        if qid == 2:
            self.type_correctness[self.question_type_dict[instance['case_id']]][1] += 1
        return False

    def get_edits_without_contamination(self, problem_case):
        """
        Inputs:
          problem_case: one multi-hop case that we want to get the set of edits that wouldn't contaminate it
        Outputs:
          nl_facts: a list of natural language edits. e.g. "John Milton is a citizen of Spain"
          triple_labeled: a list of edits in triples of text. e.g. "(John Milton, {} is a citizen of, Spain)"
          triple_ids: similar to above but in id form. E.g. "(Q79759, P27, Q29)"
          case_index: the case_id of the case that the j-th edit are in.

        NOTE: the returned values may contain duplicate edits (since an edit may come from distinct multi-hop cases).
        """

        edit_flag = problem_case['case_id'] in self.rand_list

        triples_name = 'new_triples' if edit_flag else 'orig_triples'
        correct_path = problem_case[triples_name]
        correct_path_labeled = problem_case[triples_name + '_labeled']

        nl_facts = []
        triple_labeled = []
        triple_ids = []
        case_index = []

        for d in self.dataset:
            if d['case_id'] not in self.rand_list:
                continue
            for edit in d['requested_rewrite']:
                contam_flag = False
                if any((edit['subject'] == p_labeled[0] and edit['relation_id'] == p[1] and edit['target_new_str'] != p_labeled[2]) for p, p_labeled in zip(correct_path, correct_path_labeled)):
                    contam_flag = True

                if not contam_flag:
                    nl_facts.append(
                        f'{edit["prompt"].format(edit["subject"])} {edit["target_new_str"]}')
                    triple_labeled.append(tuple(
                        [edit['subject'], edit['prompt'], edit["target_new_str"]]))
                    triple_ids.append(edit)
                    case_index.append(d['case_id'])

        return nl_facts, triple_labeled, triple_ids, case_index
