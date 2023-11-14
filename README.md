Human-Centric Autonomous Systems With LLMs for User Command Reasoning
---

[![arXiv](https://img.shields.io/badge/arXiv-TODO.TODO-b31b1b.svg)](https://arxiv.org/abs/TODO.TODO)

Task: In-Cabin User Command Understanding (UCU), [workshop in WACV2024](https://llvm-ad.github.io/challenges/)

Here is our solution code. Please check the report for more detail.

## llama & codellama

Here we introduced how to setup and the way to downloaded their model:

1. Send request to their form [Apply here in Meta official page](https://ai.meta.com/llama/) and you will receive an email with some details.

2. Dependencies: `sudo apt install wget ucommon-utils`

3. run `./scripts/llama/download.sh` to download Llama models.

4. `mamba create --name llc python=3.8 && mamba activate llc && pip install -r requirements.txt`

5. Run the example:
	```bash
	torchrun --nproc_per_node 1 scripts/main_codellama.py
	```

## GPT API

copy your OPENAI_API_KEY and save it in `.env`.
`OPENAI_API_KEY='xxxx'`

install below
```
pip install openai
pip install -U python-dotenv
```
Run the example:

```bash
python scripts/main_gpt.py --provide_few_shots True --step_by_step True
```

## Evaluation

You will have a result `.json` file finally. Then run the `eval.py` For each task-level accuracy:

```bash
python scripts/eval.py -g assets/ucu.csv -e assets/result/test.json
```

Here is the demo output:
```
Evaluating assets/result/gpt-4_best.json ...
| Task                |   Accuracy |
|---------------------+------------|
| Perception          |   0.931756 |
| In-cabin monitoring |   0.748863 |
| Localization        |   0.915378 |
| Vehicle control     |   0.88626  |
| Entertainment       |   0.944495 |
| Personal data       |   0.859873 |
| Network access      |   0.919927 |
| Traffic laws        |   0.915378 |
| Overall             |   0.890241 |
```

### LLVM_AD Official Leaderboard

Here is official evaluate with `-o`:

```bash
python3 scripts/llvm_ad/official_eval.py -o -g assets/ucu.csv -e assets/result/gpt4_best.csv
```
We attach the raw `.json` output files (with explainations and output), and its corresponding `.cvs` files (binary output only) under `assets/result` folder. 

Here is demo output:
```
Since the input file is .json, we save the prediction to .csv file at assets/result/gpt-4_best.csv
Evaluating assets/result/gpt-4_best.json ...
Following is the evaluation result in official way:

Command-level acc: 0.38034576888080074
Question-level acc: 0.8902411282984531
```

## Acknowledgements

This work was funded by Vinnova, Sweden (research grant). The computations were enabled by the supercomputing resource Berzelius provided by National Supercomputer Centre at Linköping University and the Knut and Alice Wallenberg foundation, Sweden.

This implementation is based on codes from several repositories. Thanks for these authors who kindly open-sourcing their work to the community. Please see our paper reference part to get more information.

❤️: [llvm-ad](https://llvm-ad.github.io/), [llama2](https://github.com/facebookresearch/llama/tree/main)

### Cite Our Paper
```bash
@article{yi2023drivecmdllm,
  author={Yang, Yi and Zhang, Qingwen and Li, Ci and Simões Marta, Daniel and Batool, Nazre and Folkesson, John},
  title={Human-Centric Autonomous Systems With LLMs for User Command Reasoning},
  journal={arXiv preprint arXiv:TODO.TODO},
  year={2023}
}
```