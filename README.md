# Adversarial Noisy Instruction Tuning for Enhancing NLU in Large Language Models
Codes for our paper *Adversarial Noisy Instruction Tuning for Enhancing NLU in Large Language Models*

<!-- *Visual instruction tuning towards large language and vision models with GPT-4 level capabilities.*-->

## Overview
<p align="center">
    <a> <img src=imgs/over_view.png width="100%"> </a>
</p>
Overview of Adversarial Noisy Instruction Tuning (ANIT). The example in this figure comes from the Conll03 dataset. The raw instruction and corresponding noisy instruction and their semantic distortion quantification are in the left section. The process of how the model performs adversarial training based on these instructions during the training process is in the right section.

## Contents
- [Install](#install)
- [LLMs](#LLMs)
- [Dataset](#dataset)
- [Noise Instruction build](#Noise-Instruction-build)
- [Train](#train)

## Install
Mostly refer to fire-fly installation
1. Clone this repository and navigate to project folder

2. Install Package
```Shell
conda create -n ANIT python=3.10 -y
conda activate ANIT
pip install --upgrade pip  # enable PEP 660 support
pip install -r requirements.txt
```

## LLMs
In our study, we use [ Gemma (2B)](https://huggingface.co/google/gemma-2-2b), [LLaMA2 (7B)](https://huggingface.co/meta-llama/Llama-2-7b) , and [LLaMA3 (8B)](https://huggingface.co/meta-llama/Meta-Llama-3-8B) for our experiments. These models cover the parameter ranges commonly employed in LLMs. LLaMA3 represents an improvement over LLaMA2, which is achieved through the use of expanded pre-training data and an augmented vocabulary.


## dataset
We conducted experiments on eight datasets of four tasks, all of which are public datasets. The detailed information is shown in the table:

| TASK  | Name  | Link  |
|-----------|-----------|-----------|
| NER | ConLL03 | https://drive.google.com/drive/folders/1ZxytgzPLTA7ge9sX-JgIoIZj7kUCdh9h?usp=sharing |
| NER | Ontonotes | https://catalog.ldc.upenn.edu/LDC2013T19 |
| RE | NYT | https://drive.google.com/file/d/1-3uBc_VfaCEWO2_FegzSyBXNeFmqhv7x/view |
| RE | SciERC | https://drive.google.com/drive/folders/1_u6pIe7Dw3Lqy4mF2m1UFqmKmGeM40zS?usp=sharing |
| TC | SST2 | https://huggingface.co/datasets/stanfordnlp/sst2 |
| TC | AGNews | https://huggingface.co/datasets/fancyzhx/ag_news |
| ABSA | 14Lap | https://github.com/xuuuluuu/SemEval-Triplet-data/tree/master/ASTE-Data-V2-EMNLP2020 |
| ABSA | 14Rest | https://github.com/xuuuluuu/SemEval-Triplet-data/tree/master/ASTE-Data-V2-EMNLP2020 |

## Noise Instruction build

To build the noise instructions, run the following command to get the corresponding noise instructions.

```Shell

```

## Train
The training process involves several stages to ensure optimal performance of the models:

1. Preprocessing: Clean and format the data and construct different types of noise data to suit the input requirements of the model.
2. Adversarial Noisy Instruction Tuning: Models are trained using both raw and noisy instructions to enhance their resilience to input perturbations.
To train the models, follow these steps:

```Shell
CUDA_VISIBLE_DEVICES=2 python train_qlora.py 
    --output_dir=${OUTPUT_PATH},
    --model_name_or_path=${Model_NAME},
    --train_file=${TRAIN_FILE},
    --eval_file=${EVAL_FILE},
    --num_train_epochs=10,
    --per_device_train_batch_size=16,
    --per_device_eval_batch_size=16,
    --gradient_accumulation_steps=2,
    --learning_rate=1e-4,
    --max_seq_length=512,
    --logging_steps=10,
    --evaluation_strategy=epoch,             
    --save_total_limit=10,
    --lr_scheduler_type=cosine,
    --warmup_ratio=0.04,
    --lora_rank=16,
    --lora_alpha=16,
    --lora_dropout=0.05,
    --task=${TASK_NAME},

    --gradient_checkpointing=true,
    --disable_tqdm=false,
    --optim=paged_adamw_32bit,
    --seed=42,
    --fp16=true,
    --report_to=tensorboard,
    --dataloader_num_workers=10,
    --save_strategy=epoch,
    --weight_decay=0.01,
    --max_grad_norm=0.3,
    --remove_unused_columns=false
```