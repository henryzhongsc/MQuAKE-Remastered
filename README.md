# MQuAKE-Remastered

This repository contains utilities for modifying datasets and extracting masked edits that do not contaminate a given problem case. MQuAKE-Remastered is the enhanced version of the original MQuAKE dataset proposed as a benchmark with respect to multi-hop knowledge editing tasks. We offer various ways in which unproblematic, clean, and efficient editing of data can be effected without contamination from cases that are irrelevant to the new case taking over the interest in data integrity for secure model evaluation.

## Overview

LLMs often get factual questions wrong or provide outdated answers due to the limitations of learning during training and knowledge cut-off dates. Knowledge editing is a natural candidate for this purpose because it effectively patches such errors without changing large parts of the unrelated model, but knowledge editing is intricate and highly interdependent.

The main benchmark for evaluating multi-hop knowledge editing is the original MQuAKE dataset. However, our analysis of MQuAKE reveals that up to 33% or 76% of its questions and labels may contain errors, leading to unreliable evaluations. MQuAKE-Remastered corrects these issues and thus presents a more accurate and reliable dataset.

## Key Contributions
Comprehensive audit of the MQuAKE dataset, identifying key error patterns.
Fixes and remastering of the dataset, preserving its original size and intent while addressing errors.
Re-benchmarking of all major knowledge editing methods on the fixed dataset.
Guidance on faithful evaluation to avoid overfitting to dataset-specific properties.

## Installation

Clone this repository and install the necessary dependencies:
```bash
git clone https://github.com/henryzhongsc/MQuAKE-Remastered.git
cd MQuAKE-Remastered
pip install -r requirements.txt

```


## API Overview

The `get_masked_edits` function is designed to identify and extract edits from MQuAKE without contaminating a specified problem case. 

## Get random list:
### Use case:
You can now import and use these lists of several edit nums in the main script. Here’s an example of how to do that:
```python
from edit_cases import (
    rand_list_T_1, rand_list_T_100, rand_list_T_500, rand_list_T_all,
    rand_list_3k_1, rand_list_3k_100, rand_list_3k_1000, rand_list_3k_all,
    rand_list_9k_1, rand_list_9k_1000, rand_list_9k_3000, rand_list_9k_6000, rand_list_9k_all,
    rand_list_3151_1, rand_list_3151_100, rand_list_3151_1000, rand_list_3151_all,
)

# Now you can use the imported lists
print(rand_list_T_1)
print(rand_list_T_100)
# ... and so on for other lists
```

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

## Raw Answer Dict Usage:
The purpose of using a raw answer dict is to standardize result processing. We could trace back to the llm answers and quickly look up the editing status with this object when conducting evluation.
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
### How to use cal_accuracy:
```python
from data_utils import cal_accuracy
import json

raw_answer_dict_name = 'xxx' # replace with a valid raw_answer_dict_name
with open(f"raw_answer_dict_folder/{raw_answer_dict_name}.json", 'r') as f:
  raw_answer_dict = json.load(f)

with open('datasets/modified_mquake/MQuAKE-Remastered-CF-3k.json', 'r') as f:
  dataset = json.load(f)

result, correct, total = cal_accuracy(dataset, raw_answer_dict)
```
#### Outputs:
```output (suppose caseid 10 is wrong and 11 is correct)
result = {
    'edited': 0.0,
    'unedited': 1.0
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
## Benchmarking


## CC BY 4.0 License:
MQuAKE-Remastered © 2024 is licensed under Creative Commons Attribution 4.0 International 
