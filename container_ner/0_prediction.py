import os
import torch
from fast_bert.data_cls import BertDataBunch
from fast_bert.data_ner import BertNERDataBunch
from fast_bert.learner_cls import BertLearner
from predictor_ner import NERPredictor
import time

from transformers import AutoTokenizer

import warnings

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

class BertNERPredictor(object):
    def __init__(
        self,
        model_path,
        label_path,
        model_type="bert",
        use_fast_tokenizer=True,
        do_lower_case=True,
        device=None,
    ):
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.model_path = model_path
        self.label_path = label_path
        self.model_type = model_type
        self.do_lower_case = do_lower_case
        self.device = device

        # Use auto-tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, use_fast= use_fast_tokenizer
        )

        self.learner = self.get_learner()

    def get_learner(self):
        databunch = BertNERDataBunch(
            self.label_path,
            self.tokenizer,
            train_file=None,
            val_file=None,
            batch_size_per_gpu=32,
            max_seq_length=512,
            multi_gpu=False,
            model_type=self.model_type,
            no_cache=True,
        )

        learner = NERPredictor.from_pretrained_model(
            databunch,
            self.model_path,
            device=self.device
        )

        return learner

    def predict_batch(self, texts, group=True, exclude_entities=["O"]):
        predictions = []

        for text in texts:
            pred = self.predict(text, group=group, exclude_entities=exclude_entities)
            if pred:
                predictions.append(pred)
        return predictions

    def predict(self, text, group=True, exclude_entities=["O"]):
        predictions = self.learner.predict(
            text, group=group, exclude_entities=exclude_entities
        )
        return predictions
