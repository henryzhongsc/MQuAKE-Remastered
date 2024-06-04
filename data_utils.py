def get_masked_edits(dataset, edited_cases, problem_case, edit_flag):
    """
    Inputs:
      dataset: the dataset of interest
      edited_cases: a list of caseid of edited cases
      problem_case: one multi-hop case that we want to get the set of edits that wouldn't contaminate it
      edit_flag: a boolean (True if problem_case is an edited case, False otherwise).
            Note this affects the correct path of this instance (resulting in different edits that would contaminate it)
    Outputs:
      nl_facts: a list of natural language edits. e.g. "John Milton is a citizen of Spain"
      triple_labeled: a list of edits in triples of text. e.g. "(John Milton, {} is a citizen of, Spain)"
      triple_ids: similar to above but in id form. E.g. "(Q79759, P27, Q29)", where Q79759, P27, Q29 are ids of entity
      case_index: the "caseid-1" (used for list index accessing) of the case that the j-th edit are in.

    NOTE: the returned values may contain duplicate edits (since an edit may come from distinct multi-hop cases).

    """

    if edit_flag:
        assert problem_case['case_id'] in edited_cases

    triples_name = 'new_triples' if edit_flag else 'triples'
    correct_path = problem_case['orig'][triples_name]

    nl_facts = []  # a list of natural language edits. e.g. "John Milton is a citizen of Spain"
    triple_labeled = []  # a list of edits in triples of text. e.g. "(John Milton, {} is a citizen of, Spain)"
    triple_ids = []  # similar to above but in id form. E.g. "(Q79759, P27, Q29)", where Q79759, P27, Q29 are ids of
    # entity or relation.

    case_index = []  # corresponding case index (starts from 0 for list accessing) of the edit

    for d in dataset:
        if d['case_id'] not in edited_cases:
            continue
        # want to check if d will contaminate problem_case:
        for edit, edit_extra_info in zip(d['orig']['edit_triples'], d['requested_rewrite']):
            contam_flag = False
            if any((edit[0] == p[0] and edit[1] == p[1] and edit[2] != p[2]) for p in correct_path):
                # if the edit is the same subject and relation but different answer to a specific hop -> contamination

                contam_flag = True

            # add this edit to the edit bank:
            if not contam_flag:
                nl_facts.append(
                    f'{edit_extra_info["prompt"].format(edit_extra_info["subject"])} {edit_extra_info["target_new"]["str"]}')
                triple_labeled.append(tuple(
                    [edit_extra_info['subject'], edit_extra_info['prompt'], edit_extra_info["target_new"]["str"]]))
                triple_ids.append(edit)
                case_index.append(d['case_id'] - 1)

    return nl_facts, triple_labeled, triple_ids, case_index


def process_mquake_remastered_cf_6334(dataset, edit_num = 6334): # edit_num = {100, 1000, 3000, 6334}
    train_set = []
    test_set = []

    train_set_edited_caseid = set()
    test_set_edited_caseid = set()

    for d in dataset:
      labels = d['6334_split'][edit_num]
      if 'train_edited' in labels:
        train_set.append(d)
        train_set_edited_caseid.append(d['case_id'])

      if 'test_edited_unique' in labels:
        test_set.append(d)
        test_set_edited_caseid.append(d['case_id'])
      elif 'test_edited' in labels:
        test_set.append(d)
        test_set_edited_caseid.append(d['case_id'])
      elif 'test_unedited' in labels:
        test_set.append(d)


    print(f"edit_num = {edit_num}")
    print(f"train_set size: {len(train_set)}")
    print(f"test_set size: {len(test_set)}")

    return train_set, test_set, train_set_edited_caseid, test_set_edited_caseid
    # test_edited includes cases that are: 1) edited and unique to test_set; 2) unedited and unique to test_set; and 3) edited, exist in both train_set & test_set; such type of cases are a subset of train_edited
    # In practice, one can just grab the whole test_set and conduct normal evaluation, where for each case, you first check if this case is in test_set_edited_caseid, then feed it accordingly. Do make sure you register the edit status of each case accordingly in your raw output.


def check_answer(edit_flag, instance, ans):
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

    # Return true if ans matches the answer or any of the aliases
    return ans_upper == instance_answer_upper or ans_upper in instance_answer_alias_upper


# Evaluation:
def cal_accuracy(dataset, raw_answer_dict, use_6334=False):
    if not use_6334:
        acc_list = ["unedited_acc", "edited_acc"]
    else:
        acc_list = ["unedited_acc", "train_edited_acc", "test_train_overlap_edited_acc", "test_unique_edited_acc"]

    total = {}
    correct = {}
    for acc in acc_list:
        total[acc] = set()
        correct[acc] = set()

    for d in dataset:
        if d['case_id'] not in raw_answer_dict.keys():
            continue
        edited_flag = raw_answer_dict[d['case_id']]['edited']
        this_is_correct = any(check_answer(edited_flag, d, ans) for ans in raw_answer_dict[d['case_id']]['answers'])
        caseid = d['case_id']
        if use_6334:
            labels_6334 = d['6334_split']
            if 'train_edited' in labels_6334:
                total['train_edited_acc'].add(caseid)
                if this_is_correct:
                    correct['train_edited_acc'].add(caseid)

            if 'test_edited_unique' in labels_6334:
                total['test_unique_edited_acc'].add(caseid)
                if this_is_correct:
                    correct['test_unique_edited_acc'].add(caseid)
            else:
                if 'test_edited' in labels_6334:
                    total['test_train_overlap_edited_acc'].add(caseid)
                    if this_is_correct:
                        correct['test_train_overlap_edited_acc'].add(caseid)

            if 'test_unedited' in labels_6334:
                total['unedited_acc'].add(caseid)
                if this_is_correct:
                    correct['unedited_acc'].add(caseid)

        else:
            if edited_flag:
                total['edited_acc'].add(caseid)
                if this_is_correct:
                    correct['edited_acc'].add(caseid)
            else:
                total['unedited_acc'].add(caseid)
                if this_is_correct:
                    correct['unedited_acc'].add(caseid)

    result = {}
    for acc in acc_list:
        result[acc] = len(correct[acc]) / len(total[acc]+ 1e-8)

    return result, correct, total
