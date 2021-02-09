import random
import logging

import torch
import numpy as np

from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from model import ElectraForMultiLabelClassification

from transformers import (
    ElectraConfig,
    ElectraTokenizer,
)

CONFIG_CLASSES = {
    "koelectra-small-v1": ElectraConfig,
    "koelectra-base-v1": ElectraConfig,
    "koelectra-small-v3": ElectraConfig,
    "koelectra-base-v3": ElectraConfig
}

TOKENIZER_CLASSES = {
    "koelectra-small-v1": ElectraTokenizer,
    "koelectra-base-v1": ElectraTokenizer,
    "koelectra-small-v3": ElectraTokenizer,
    "koelectra-base-v3": ElectraTokenizer
}

MODEL_CLASSES = {
    "koelectra-small-v1": ElectraForMultiLabelClassification,
    "koelectra-base-v1": ElectraForMultiLabelClassification,
    "koelectra-small-v3": ElectraForMultiLabelClassification,
    "koelectra-base-v3": ElectraForMultiLabelClassification
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


def compute_metrics(labels, preds):
    assert len(preds) == len(labels)
    results = dict()

    results["accuracy"] = accuracy_score(labels, preds)
    results["macro_precision"], results["macro_recall"], results[
        "macro_f1"], _ = precision_recall_fscore_support(
        labels, preds, average="macro")
    results["micro_precision"], results["micro_recall"], results[
        "micro_f1"], _ = precision_recall_fscore_support(
        labels, preds, average="micro")
    results["weighted_precision"], results["weighted_recall"], results[
        "weighted_f1"], _ = precision_recall_fscore_support(
        labels, preds, average="weighted")

    return results
