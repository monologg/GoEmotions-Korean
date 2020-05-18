import os
import random
import logging

import torch
import numpy as np

from model import (
    ElectraForMultiLabelClassification,
)

from transformers import (
    ElectraConfig,
    ElectraTokenizer,
)

CONFIG_CLASSES = {
    "koelectra-small": ElectraConfig,
    "koelectra-base": ElectraConfig
}

TOKENIZER_CLASSES = {
    "koelectra-small": ElectraTokenizer,
    "koelectra-base": ElectraTokenizer
}

MODEL_CLASSES = {
    "koelectra-small": ElectraForMultiLabelClassification,
    "koelectra-base": ElectraForMultiLabelClassification
}


def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def simple_accuracy(labels, preds):
    return (labels == preds).mean()


def acc_score(labels, preds):
    return {
        "acc": simple_accuracy(labels, preds),
    }


def compute_metrics(task_name, labels, preds):
    assert len(preds) == len(labels)
    if task_name == "intent-cls":
        return acc_score(labels, preds)
    else:
        raise KeyError(task_name)
