"""
The script is for finetuning large language models
"""

# Set project root dir
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Start importing modules
from src.trainer import WeightedTrainer, compute_metrics
from src.dataloader import DataLoader
from src.utils import load_config, display_hyperparams

from typing import Union, Dict, Iterable
from pathlib import Path
from argparse import ArgumentParser

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments
    )

from peft import (
    LoraConfig,
    TaskType,
    get_peft_model
)

from datasets import Dataset
import evaluate

import numpy as np


def set_train_args(config: Dict) -> TrainingArguments:
    out_dir = Path(config['cache_dir']) / f"ft_{config['model_name']}"
    model_dir = Path(f"lora_{config['model_name']}_lr{config['lr']}_bsz{config['batch_size']}")

    training_args = TrainingArguments(
        output_dir=out_dir / model_dir,
        logging_steps=500,
        learning_rate=config['lr'],
        weight_decay=0.001,
        per_device_train_batch_size=config['batch_size'],
        per_device_eval_batch_size=config['batch_size'],
        num_train_epochs=config['epochs'],
        eval_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model='eval_F1',
        greater_is_better=True,
        disable_tqdm=False,
        report_to='none'
    )

    return training_args


def get_args():
    parser = ArgumentParser()
    parser.add_argument('-C', '--configFile', required=True, help='Configuration file as YAML format')
    parser.add_argument('-D', '--data', required=True, help='The directory path of dataset')
    return parser.parse_args()

if __name__ == '__main__':

    # load external options
    args = get_args()
    data_dir = args.data
    configs = load_config(args.configFile)
    display_hyperparams('Finetune Overview', configs)

    # -------------------------------
    # |     Data preprocessing      |
    # -------------------------------

    # Step 1:
    # ----------
    # Initialize tokenizer and collator objects

    checkpoint = configs['checkpoint']
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # (Optional)
    # ----------
    # For decoder-only models, explicitly assigan <EOS> to <PAD>
    if configs['decoder_only']:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token


    # Step 2:
    # ----------
    # Load raw dataset and represent them as HF's `Dataset` object

    dataset = DataLoader().load(data_dir=data_dir, tokenizer=tokenizer)

    # -------------------------------
    # |        Model config         |
    # -------------------------------

    # Step 3:
    # ----------
    # LoRA config

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=3,
        lora_alpha=16,
        lora_dropout=0.1
    )

    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
    model = get_peft_model(model, peft_config)

    # (Optional): for decoder-only models
    if configs['decoder_only']:
        model.config.pad_token_id = model.config.eos_token_id

    # Step 4:
    # ----------
    # training arguments config

    # cache_dir = Path.cwd() / '.cache'
    # cache_dir.mkdir(parents=True, exist_ok=True)
    train_args = set_train_args(configs)

    # TODO:
    # for arg in train_args:
    #   trainer = ...

    trainer = WeightedTrainer(
        model=model,
        args=train_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['dev'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )


    trainer.train()






