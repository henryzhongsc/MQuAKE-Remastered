# MQuAKE-Remastered

This repository contains utilities for modifying datasets and extracting masked edits that do not contaminate a given problem case.

## Overview

The `get_masked_edits` function is designed to identify and extract edits from MQuAKE without contaminating a specified problem case. 

## Function Usage

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
