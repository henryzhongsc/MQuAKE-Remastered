# MQuAKE-Remastered

## Introduction

MQuAKE-Remastered is a **corrected and enhanced version** of the original MQuAKE dataset, designed to provide a more reliable benchmark for evaluating multi-hop knowledge editing in large language models (LLMs). The original [MQuAKE](https://github.com/princeton-nlp/MQuAKE) dataset contained **critical errors**, including edit contamination, missing information in question instructions, conflicting edits, and duplicate cases. These issues affected the realiability of evaluation results, making it difficult to assess LLMs' true performance in knowledge editing.

MQuAKE-Remastered preserves the original dataset's scale and structure while introducing key improvements, ensuring it serves as a robust, contamination-free benchmark for LLMs in counterfactual and temporal knowledge editing tasks.

The dataset consists of four splits:
- **`MQuAKE-Remastered-CF-3k`**: A counterfactual dataset (subset of `CF-9k`) that fixes the original errors while maintaining the same size as `MQuAKE-CF-3k`.
- **`MQuAKE-Remastered-CF-9k`**: A full-scale counterfactual dataset that corrects all errors in `MQuAKE-CF`.
- **`MQuAKE-Remastered-CF-6334`**: A structured subset of `CF-9k` with dedicated **training and testing splits**, suitable for **parameter-based** editing methods.
- **`MQuAKE-Remastered-T`**: A temporal dataset designed for evaluating knowledge updates based on real-world factual changes.

We are also presenting the replication code for `gwalk` and `mello` in this repo.

## Try GWalk on Google Colab

We provide a Google Colab notebook where you can **try out GWalk**, our **minimally invasive, state-of-the-art knowledge editing method** for MQuAKE-Remastered. GWalk is a performant and faithful method for evaluating knowledge editing performance.

You can access the Colab page here: **[Link to be provided]**

## Replication Guide

To replicate our results, follow these steps:

### 1. Install Dependencies

In our setting, we used **Python 3.8.6** installed, but other python version should also work. Then, install the required dependencies using:

```bash
pip install -r requirements.txt
```

### 2. Run GWalk for CF-3k

To replicate our results using GWalk on **Meta-Llama-3-8B**, execute the following script:

```bash
./scripts/CF-3k/gwalk/Meta-Llama-3-8B.sh
```

This script will run GWalk on the **CF-3k** dataset using the **Meta-Llama-3-8B model**. Note: you can edit the `edit_num` in `Meta-Llama-3-8B.sh`, where `edit_num` stands for the number of cases that are considered as edited cases. 
