# MQuAKE-Remastered

This repository contains utilities for modifying datasets and extracting masked edits that do not contaminate a given problem case.

## Overview

The `get_masked_edits` function is designed to identify and extract edits from MQuAKE without contaminating a specified problem case. 

## get_masked_edits Method Usage

### Inputs:
- `dataset`: The dataset of interest.
- `rand_list`: A list of case IDs of edited cases.
- `problem_case`: A multi-hop case for which we want to obtain a set of edits that wouldn't contaminate it.
- `edit_flag`: A boolean indicating whether the `problem_case` is an edited case.

### Outputs:
- `nl_facts`: A list of natural language edits (e.g., "John Milton is a citizen of Spain").
- `triple_labeled`: A list of edits in triples of text (e.g., "(John Milton, {} is a citizen of, Spain)").
- `triple_ids`: Similar to above but in ID form (e.g., "(Q79759, P27, Q29)").
- `case_index`: The "caseid-1" (used for list index accessing) of the case that the j-th edit is in.

### Example Usage

#### Sample 1000 Case IDs

```python
import random

# Sample 1000 case_ids out of the 3000-long MQuAKE-CF-3k-Remastered
rand_list = random.sample(range(1, len(dataset_modifying)+1), 1000)

new_facts = set()
for d in dataset_modifying:
    if d['case_id'] not in rand_list:
        continue
    for r in d["requested_rewrite"]:
        new_facts.add(f'{r["prompt"].format(r["subject"])} {r["target_new"]["str"]}')
new_facts = list(new_facts)
print(len(new_facts))

```
```output
1129
```
#### Use Case 1: Extracting Non-Contaminating Edits

```python
for i, d in enumerate(dataset_modifying):
    nl_facts, triple_labeled, triple_ids, case_index = get_masked_edits(dataset_modifying, rand_list, d, edit_flag=d['case_id'] in rand_list)
    print(f"len(nl_facts) = {len(set(nl_facts))}")

    if i == 2:
        break
```
```output
len(nl_facts) = 1128 (One outside contamination will affect the correct path so we mask the fact out)
len(nl_facts) = 1129
len(nl_facts) = 1129
```

#### Use Case 2: Specific Instance Extraction

```python
d = dataset_modifying[0]
nl_facts, triple_labeled, triple_ids, case_index = get_masked_edits(dataset_modifying, rand_list, d, edit_flag=d['case_id'] in rand_list)

print(f"triple_labeled[0] = {triple_labeled[0]}")
print(f"triple_ids[0] = {triple_ids[0]}")
print(f"case_index[0] = {case_index[0]}")


```

```output
triple_labeled[0] = ('United States of America', 'The capital of {} is', 'El Campu')
triple_ids[0] = ['Q30', 'P36', 'Q3775140']
case_index[0] = 2 (case_id of 3)
```

## raw answer dict
An example of raw answer dict where 10 and 11 are caseids:
```output
raw_answer_dict = {
    10: {
        'edited': True,
        'answers': [
            'answer1',
            'answer2',
            'answer3'
        ]
    },
    11: {
        'edited': False,
        'answers': [
            'Europe',
            'answer5',
            'answer6'
        ]
    }
}
```
### How to calculate ACC:
```python
def check_answer(edit_flag, instance, ans):
    # multi-hop accuracy:
    answer = "answer"
    answer_alias = "answer_alias"
    if edit_flag:
        answer = "new_" + answer
        answer_alias = "new_" + answer_alias
    
    return ans == instance[answer] or ans in instance[answer_alias]


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
        result[acc] = len(correct[acc]) / len(total[acc]) if len(total[acc]) != 0 else None
        
    return result, correct, total
```
```output (suppose caseid 10 is wrong and 11 is correct)
result = {
    'edited': 0,
    'unedited': 1
},
correct = {
    'edited' = set(),
    'unedited' = set([11])
},
total = {
    'edited' = set([10]),
    'unedited' = set([11])
}
```
