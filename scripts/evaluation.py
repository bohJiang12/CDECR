"""
The script is for evaluating a single fine-tuned model on the test set
w/ different hyperparameter settings
"""
# Set project root dir
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import *
from src.dataloader import build_dataset_from
from src.trainer import *

from typing import Dict
from argparse import ArgumentParser
from pathlib import Path

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer
)

import numpy as np


def get_args():
    parser = ArgumentParser()
    parser.add_argument('-M', '--model', required=True, help="Name of the model to evaluate (either 'roberta' or 'llama')")
    parser.add_argument('--decoderOnly', required=True, help="Option for indicating if the model is decoder-only model")
    return parser.parse_args()


def eval_model_from(model_dir: Path | str,
                    test_path: Path | str,
                    out_dir: Path | str,
                    decoder_only: bool) -> Dict:
    """Evaluate a model on the test set given its path

    Args:
        model_dir: path object of the model to be loaded
        testset: path object of the test set
        out_dir: dest. path where the eval results to be saved
        decoder_only: indicate if the model to be evaluated is decoder-only

    Returns:
        Dict: a dictionary of metric scores including accuracy, precision, and f1

    Usage:
        >>> from pathlib import Path
        >>> model_path = Path().parent / ".cache/ft_roberta/lora_roberta_lr0.0003_bsz8/checkpoint-28416"
        >>> eval_model_from(model_path)
        {'acc': 0.937, 'precision': 0.713, 'f1': 0.642}
    """
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    if decoder_only:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    testset = build_dataset_from(test_path, tokenizer)

    train_args = TrainingArguments(
        output_dir=out_dir,
        do_train=False,
        do_eval=True,
        per_device_eval_batch_size=16
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        eval_dataset=testset,
        compute_metrics=compute_metrics,
        data_collator=data_collator
    )

    return trainer.evaluate()


if __name__ == '__main__':
    # Set up directory path and obtain script arguments
    args = get_args()
    model_name = args.model
    decoder_only = args.decoderOnly

    root_dir = set_wkdir_as_root(Path.cwd())
    out_dir = root_dir / '.out'
    testset_path = root_dir / 'data/event_pairs.test'
    model_dir = root_dir / f".cache/ft_{model_name}"

    # Start evaluating models under different settings
    report = {}

    for dir in model_dir.iterdir():
        if dir.is_dir():
            model_config = dir.stem + dir.suffix
            model_path = list(dir.glob('checkpoint-*'))[0]  # assume only the best checkpoint is selected
            res = eval_model_from(model_dir=model_path,
                                  test_path=testset_path,
                                  out_dir=out_dir,
                                  decoder_only=decoder_only)

            report.update(
                {model_config:
                    {'acc': '%.3f' % res['eval_acc']['accuracy'],
                        'precision': '%.3f' % res['eval_precision']['precision'],
                        'f1': '%.3f' % res['eval_F1']
                    }
                }
            )

            update_report(report_path=root_dir/'report.json', new_data=report)
