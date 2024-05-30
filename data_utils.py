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


def process_mquake_remastered_cf_6334(dataset, edit_num = 6334):
    train_edited = []
    test_unedited = []
    test_edited = []
    test_edited_overlap = [] # intersection of train_edited and test_edited

    train_edited_caseid = set()
    test_edited_caseid = set()



    for d in dataset:
      labels = d['6334_split'][edit_num]
      if 'train_edited' in labels:
        train_edited.append(d)
        train_edited_caseid.add(d['case_id'])

      if 'test_edited_unique' in labels:
        test_edited_overlap.append(d)
      else:
        if 'test_edited' in labels:
          test_edited.append(d)
          test_edited_caseid.add(d['case_id'])

      if 'test_unedited' in labels:
        test_unedited.append(d)


    print("edit_num = ", edit_num)
    print("train_edited: ", len(train_edited))
    print("test_unedited: ", len(test_unedited))
    print("test_edited: ", len(test_edited))
    print("test_edited_overlap: ", len(test_edited_overlap))

    return train_edited, test_unedited, test_edited, test_edited_overlap
    # parameter-based methods should train on train_edited then separately evaluate on test_unedited and test_edited with three separate acc reported.
    # test_overlaped is included in test_edited, so no need to separatly
