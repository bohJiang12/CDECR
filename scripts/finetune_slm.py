"""
This is the script for fine-tuning (small) language models
"""
from src.events import EventPair

from typing import Union, Dict, Iterable
from pathlib import Path
from argparse import ArgumentParser
import yaml


from transformers import (
    RobertaForSequenceClassification,
    RobertaTokenizer,
    DataCollatorWithPadding,
    Trainer,
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

checkpoint = 'FacebookAI/roberta-base'
tokenizer = RobertaTokenizer.from_pretrained(checkpoint)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# -------------------------------
# |     Data preprocessing      |
# -------------------------------

def load_data(fpath: Union[str, Path]) -> Iterable[EventPair]:
    """Load and yield dataset from its path"""
    is_test_set = Path(fpath).suffix == '.test'
    with open(fpath, 'r', encoding='utf-8') as f:
        for line in f:
            yield EventPair(line, is_test_set)


def tokenize_func(sample):
    """Define tokenization function for a single sample"""
    return tokenizer(
        sample['event_1'],
        sample['event_2'],
        truncation=True
    )


def build_dataset_from(fpath) -> Dataset:
    """Build HF's `Dataset` for training"""
    event_pairs = load_data(fpath)
    events_1, events_2, labels = [], [], []

    for pair in event_pairs:
        event_1, event_2 = pair.events
        label = pair.label

        events_1.append(event_1)
        events_2.append(event_2)
        labels.append(label)

    data_dict = {
        'event_1': events_1,
        'event_2': events_2,
        'label': labels
    }

    dataset = Dataset.from_dict(data_dict)

    return dataset.map(tokenize_func, batched=True)


# -------------------------------
# |         Metric              |
# -------------------------------

def compute_metrics(eval_preds):
    metric = evaluate.load('accuracy')
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(references=labels, predictions=predictions)

# -------------------------------
# |  Script args & load configs |
# -------------------------------

def get_args():
    parser = ArgumentParser()
    parser.add_argument('-C', '--configFile', required=True, help='Configuration file as YAML format')
    parser.add_argument('-D', '--data', required=True, help='The directory path of dataset')
    return parser.parse_args()


def load_config(config_file):
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    return config


if __name__ == '__main__':

    args = get_args()
    config = load_config(args.configFile)
    model_name = f"lora_roberta_lr{config['lr']}_bsz{config['batch_size']}"


    data_dir = Path(args.data)
    train_data = build_dataset_from(data_dir / "event_pairs.train")
    dev_data = build_dataset_from(data_dir / "event_pairs.dev")
    test_data = build_dataset_from(data_dir / "event_pairs.test")

    # -------------------------------
    # |         LoRA config         |
    # -------------------------------

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=3,
        lora_alpha=32,
        lora_dropout=0.1
    )

    # -------------------------------
    # |       Trainer config        |
    # -------------------------------

    cache_dir = Path(config['cache_dir'])
    cache_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=cache_dir / model_name,
        learning_rate=config['lr'],
        per_device_train_batch_size=config['batch_size'],
        per_device_eval_batch_size=config['batch_size'],
        num_train_epochs=config['epochs'],
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        disable_tqdm=False,
        report_to='none'
    )

    # -------------------------------
    # |         Model config        |
    # -------------------------------

    model = RobertaForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
    model = get_peft_model(model, peft_config)

    # -------------------------------
    # |         Trainer config      |
    # -------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=dev_data,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # start training and save the model when it's done
    trainer.train()

    model.save_pretrained(cache_dir)
