# MQuAKE-Remastered

This is the official code repository for the paper [**MQuAKE-Remastered: Multi-Hop Knowledge Editing Can Only Be Advanced with Reliable Evaluations**](https://openreview.net/forum?id=m9wG6ai2Xk).

We audit the MQuAKE benchmark — the primary evaluation suite for multi-hop knowledge editing — and find that 33–76% of its questions and labels are corrupted due to unintentional clerical or procedural oversights. We release **MQuAKE-Remastered**, a corrected version of the dataset, along with **G-Walk**, a simple graph-walk-based approach for multi-hop knowledge editing that does not exploit the quirks of the original dataset.

The corrected dataset is hosted on HuggingFace: [`henryzhongsc/MQuAKE-Remastered`](https://huggingface.co/datasets/henryzhongsc/MQuAKE-Remastered).

## Setup

```bash
uv sync
```

Then set your HuggingFace access token in `configs/global_setting.py`:
```python
hf_access_token = "hf_your_token_here"
```

## Running Experiments

All experiments are launched via `pipeline/main.py` with three config files specifying the method, dataset split, and output structure.

**Example — G-Walk with Mistral-7B on CF-3k (1000 edits):**
```bash
bash scripts/Mistral-7B-Instruct-v0.2/gwalk/CF-3k.sh <output_dir>
```

**Or run directly:**
```bash
uv run python pipeline/main.py \
    --exp_desc Mistral-7B__gwalk__CF-3k_1000 \
    --pipeline_config_dir configs/pipeline_config/Mistral-7B-Instruct-v0.2/gwalk.json \
    --eval_config_dir configs/eval_config/mquake_remastered/CF-3k_1000.json \
    --management_config_dir configs/management_config/default.json \
    --output_folder_dir <output_dir>
```

Supported methods: `gwalk`, `mello`. Supported models: `Mistral-7B-Instruct-v0.2`, `Llama-2-7b-hf`. Scripts and configs for all combinations are provided under `scripts/` and `configs/pipeline_config/`.

## Project Structure

```
configs/
  pipeline_config/<model>/         gwalk.json, mello.json
  eval_config/mquake_remastered/   CF-3k_1000.json
  management_config/               default.json
  global_setting.py                Seed, timezone, HF token

pipeline/
  main.py                          Entry point
  inference.py                     LLM loading (device_map="auto")
  inference_mquake.py              Contriever, stopping criteria, REL2SUBQ
  gwalk/                           G-Walk eval orchestrator + loop
  mello/                           MeLLo eval orchestrator + loop

eval/mquake_remastered/            Dataset class, utilities, KG operations
data/mquake_remastered/prompts/    Prompt templates (5 files)
scripts/<model>/<method>/          Bash launch scripts
utils/                             Logging, config registration, seed locking
```

## Results

After an experiment concludes, the output folder contains:
- `output.json` — configs + reported metrics under `processed_results`
- `raw_results/raw_results.json` — per-instance answers and correctness
- `exp.log` — real-time experiment log
- `backup/` — code snapshot at experiment time
- `input_configs/` — carbon copies of input configs

## Citation

```bibtex
@inproceedings{zhong2025mquake,
    title={MQuAKE-Remastered: Multi-Hop Knowledge Editing Can Only Be Advanced with Reliable Evaluations},
    author={Shaochen Zhong and Yifan Lu and Lize Shao and Bhargav Bhushanam and Xiaocong Du and Louis Feng and Yixin Wan and Yucheng Shi and Daochen Zha and Yiwei Wang and Ninghao Liu and Kaixiong Zhou and Shuai Xu and Vipin Chaudhary and Xia Hu},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025},
    url={https://openreview.net/forum?id=m9wG6ai2Xk}
}
```
