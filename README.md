# llcommand-AD

Large Language Command in Autonomous driving.

Task: In-Cabin User Command Understanding (UCU), [workshop in WACV2024](https://llvm-ad.github.io/challenges/)

Here is our solution code. Please check the report for more detail.

## llama

[Pretrained model from Meta](https://ai.meta.com/llama/) and code from [codellama](https://github.com/facebookresearch/codellama/tree/main), Here we show how to downloaded their model

1. Send request to their form and you will receive an email with some details.
2. `git clone TBD && cd llc` 
3. Dependencies: `sudo apt install wget ucommon-utils`
4. run `./download.sh` Then enter **<u>the link</u>** you received at first step.

5. `mamba create --name llc python=3.8 && mamba activate llc && pip install -e .`

6. run the example:

   ```bash
   torchrun --nproc_per_node 1 scripts/example_instructions.py \
       --ckpt_dir model/CodeLlama-7b/ \
       --tokenizer_path model/CodeLlama-7b/tokenizer.model \
       --max_seq_len 128 --max_batch_size 4
   ```

Here is the table to show how many memory we need use when run different model.

TODO table here

| Model                  | GPU Memory Cost | Running Time |
| ---------------------- | --------------- | ------------ |
| CodeLlama-7b           |                 |              |
| CodeLlama-7b-Instruct  |                 |              |
| CodeLlama-7b-Python    |                 |              |
| CodeLlama-13b          |                 |              |
| CodeLlama-13b-Instruct |                 |              |
| CodeLlama-13b-Python   |                 |              |
| CodeLlama-34b          |                 |              |
| CodeLlama-34b-Instruct |                 |              |
| CodeLlama-34b-Python   |                 |              |



## Command Analysis 

Now we will come to the challenge task.

- Data preparation: Already downloaded to this repo inside [assets](assets/ucu.csv).
- Scripts which is based on Different models require different model-parallel (MP) values [7B]
  ```bash
  torchrun --nproc_per_node 1 scripts/main_llc.py
  torchrun --nproc_per_node 2 scripts/main_llc.py
  torchrun --nproc_per_node 4 scripts/main_llc.py
  ```
- Prompt modified inside [scripts/prompt.py](scripts/prompt.py)

After run this script, we will have following print in terminal:
```
Saving results....
| Task                |   Accuracy |
|---------------------+------------|
| Perception          |   0.583333 |
| In-cabin monitoring |   0.75     |
| Localization        |   0.583333 |
| Vehicle control     |   0.583333 |
| Entertainment       |   0.833333 |
| Personal data       |   0.583333 |
| Network access      |   0.666667 |
| Traffic laws        |   0.75     |
| Overall             |   0.666667 |
```

Here is demo image:
![](assets/readme/demo.png)
