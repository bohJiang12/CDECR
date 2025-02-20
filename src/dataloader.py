# Set project root dir
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.events import EventPair

from typing import Any, Union, Dict, Iterable
from pathlib import Path

from datasets import Dataset

DATA_PREFIX = 'event_pairs'


class DataLoader:
    """Class for building HF dataset including train/dev/test"""
    def load(self, data_dir: str, tokenizer):
        dataset = {}

        for file in Path(data_dir).iterdir():
            if file.stem == DATA_PREFIX:
                key = file.suffix[1:]
                val = build_dataset_from(file, tokenizer)
                dataset.update({key: val})

        return dataset


def gen_event_pairs(fpath: str) -> Iterable[EventPair]:
    """Load and yield dataset from its path"""
    is_test_set = Path(fpath).suffix == '.test'
    with open(fpath, 'r', encoding='utf-8') as f:
        for line in f:
            yield EventPair(line, is_test_set)


def build_dataset_from(fpath, tokenizer) -> Dataset:
    """Build HF's `Dataset` for training"""

    def tokenize_func(sample):
        """Define tokenization function for a single sample"""
        return tokenizer(
            sample['event_1'],
            sample['event_2'],
            truncation=True
        )

    event_pairs = gen_event_pairs(fpath)
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
