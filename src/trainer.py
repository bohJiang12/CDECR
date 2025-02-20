from collections import Counter

import numpy as np

import torch
from torch.nn import CrossEntropyLoss

from transformers import Trainer
from datasets import Dataset
import evaluate


def compute_metrics(eval_preds):
    acc_metric = evaluate.load('accuracy')
    f1_metric = evaluate.load('f1')
    precision_metric = evaluate.load("precision")

    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    f1 = f1_metric.compute(references=labels, predictions=predictions)['f1']
    acc = acc_metric.compute(references=labels, predictions=predictions)
    precision = precision_metric.compute(references=labels, predictions=predictions)

    # return metric.compute(references=labels, predictions=predictions)
    return {'acc': acc, 'precision': precision, 'F1': f1}


def weighted_loss_fn(training_data: Dataset):
    label_counts = Counter(sample['label'] for sample in training_data)

    total_samples = sum(label_counts.values())
    class_counts = np.array([label_counts.get(0, 1), label_counts.get(1, 1)])

    weights = total_samples / (2.0 * class_counts)
    weights_tensor = torch.tensor(weights, dtype=torch.float32).to('cuda')

    return CrossEntropyLoss(weight=weights_tensor)


class WeightedTrainer(Trainer):
    def __init__(self, *args, loss_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = loss_fn if loss_fn is not None else CrossEntropyLoss()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")  # Extract labels
        outputs = model(**inputs)
        logits = outputs.logits  # Extract model logits

        loss = self.loss_fn(logits, labels)  # Compute loss

        return (loss, outputs) if return_outputs else loss