# llcommand-AD

Large Language Command in Autonomous driving.

Task: In-Cabin User Command Understanding (UCU), [workshop in WACV2024](https://llvm-ad.github.io/challenges/)

Here is our solution code. Please check the report for more detail.

## llama & codellama

Here we introduced how to setup. [Pretrained model from Meta](https://ai.meta.com/llama/) and code from [codellama](https://github.com/facebookresearch/codellama/tree/main). check the [assets/slurm](assets/slurm) for more detail on all our experiments includes different models and all ablation study running. 

Here we show how to downloaded their model:

1. Send request to their form and you will receive an email with some details.
2. Dependencies: `sudo apt install wget ucommon-utils`
3. run `./scripts/llama/download.sh` to download Llama models.
4. `mamba create --name llc python=3.8 && mamba activate llc && pip install -r requirements.txt`
5. run the example.

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

## Command Analysis 

Now we will come to the challenge task.

- Data preparation: Already downloaded to this repo inside [assets](assets/ucu.csv).
- Prompt modified inside [scripts/prompt.py](scripts/prompt.py)
- Run with `torchrun --nproc_per_node 1 scripts/main_x.py`
- Result txt and npy will be saved inside [assets/results](assets/results) so you can run it again.

Then you will have a result `.json` file finally. Then run the `eval.py` For each task-level accuracy:

```bash
python scripts/eval.py -g assets/ucu.csv -e assets/result/test.json
```

### LLVM_AD Official Leaderboard

Here is [official evaluate.py](), we copy directly from their repo but you can either input the `.json` or `.csv` file they required. 

```bash
python3 scripts/llvm_ad/official_eval.py -g assets/ucu.csv -e assets/result/test.json
```
We attach the raw `.json` output files (with explainations and output), and its corresponding `.cvs` files (binary output only) under `assets/result` folder. 

Here is demo output:
```
Since the input file is .json, we save the prediction to .csv file: assets/result/gpt4_best.csv

Following is the evaluation result in official way: 

Command-level acc: 0.38034576888080074
Question-level acc: 0.8902411282984531
```
