# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license.

import os
import torch
import json
import numpy as np
from azureml.core.model import Model
from fast_bert.data_cls import BertDataBunch
from fast_bert.data_ner import BertNERDataBunch
from fast_bert.learner_cls import BertLearner
from fast_bert.data_ner import BertNERDataBunch
from torch import nn
from typing import Dict, List, Optional, Tuple
from fast_bert.learner_util import Learner
import nltk
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    EvalPrediction,
    HfArgumentParser,
    logging
)

import warnings

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

def load_model(databunch, pretrained_path, finetuned_wgts_path, device):

    model_type = databunch.model_type
    model_state_dict = None

    if torch.cuda.is_available():
        map_location = lambda storage, loc: storage.cuda()
    else:
        map_location = "cpu"

    if finetuned_wgts_path:
        model_state_dict = torch.load(finetuned_wgts_path, map_location=map_location)
    else:
        model_state_dict = None

    config = AutoConfig.from_pretrained(
        str(pretrained_path),
        num_labels=len(databunch.labels),
        model_type=model_type,
        id2label=databunch.label_map,
        label2id={label: i for i, label in enumerate(databunch.labels)},
    )

    model = AutoModelForTokenClassification.from_pretrained(
        str(pretrained_path), config=config, state_dict=model_state_dict
    )

    model.eval()
    return model

class NERPredictor():
    @staticmethod
    def from_pretrained_model(
        databunch,
        pretrained_path,
        device,
        finetuned_wgts_path=None,
    ):
        model = load_model(databunch, pretrained_path, finetuned_wgts_path, device)
        return NERPredictor(
            databunch,
            model,
            device,
        )

    def __init__(self, data: BertNERDataBunch, 
                       model: nn.Module,
                       device):
        self.model = model
        self.label_list = data.labels
        self.tokenizer = data.tokenizer
        self.device = device

    def predict(self, text, group=True, exclude_entities=["O"]):
        if exclude_entities is None:
            exclude_entities = []

        words = nltk.word_tokenize(text)
        enc_words = []
        valid_tokens = []
        count = 0
        for word in words:
            enc_word = self.tokenizer.encode(word)[1:-1]
            valid_tokens.extend([count] + [-1] * (len(enc_word)-1))
            count += 1
            if len(enc_words) + len(enc_word) + 2 > self.tokenizer.max_len:
                break
            enc_words.extend(enc_word)

        enc_words.insert(0,self.tokenizer.cls_token_id)
        enc_words.append(self.tokenizer.sep_token_id)
        inputs = torch.tensor([enc_words])

        inputs = inputs.to(self.device)

        model = self.model.to(self.device)

        with torch.no_grad():
            outputs = model(inputs)[0]
            
            outputs = outputs.softmax(dim=2)
            predictions = torch.argmax(outputs, dim=2)

            preds = [{"index": index,  
                    "word": words[index], 
                    "entity": self.label_list[prediction], 
                    "score":output[prediction]} 
                    for output, prediction, index in zip(outputs[0].tolist()[1:-1], predictions[0].tolist()[1:-1], valid_tokens) if index >= 0]

        if group is True:
            preds = group_entities(preds)

        out_preds = []
        for pred in preds:
            if pred["entity"] not in exclude_entities:
                try:
                    pred["entity"] = pred["entity"].split("-")[1]
                except Exception:
                    pass

                out_preds.append(pred)

        return out_preds
   
def group_sub_entities(entities) -> dict:
    """
    Returns grouped sub entities
    """
    # Get the first entity in the entity group
    entity = entities[0]["entity"]
    scores = np.mean([entity["score"] for entity in entities])
    tokens = [entity["word"] for entity in entities]

    entity_group = {
        "entity": entity,
        "score": np.mean(scores),
        "word":" ".join(tokens) ,   
    }
    return entity_group

def group_entities(entities: List[dict]) -> List[dict]:
    """
    Returns grouped entities
    """

    entity_groups = []
    entity_group_disagg = []

    if entities:
        last_idx = entities[-1]["index"]

    for entity in entities:
        is_last_idx = entity["index"] == last_idx
        if not entity_group_disagg:
            entity_group_disagg += [entity]
            if is_last_idx:
                entity_groups += [group_sub_entities(entity_group_disagg)]
            continue

        # If the current entity is similar and adjacent to the previous entity, append it to the disaggregated entity group
        # The split is meant to account for the "B" and "I" suffixes
        if ((entity["entity"].split("-")[-1]
            == entity_group_disagg[-1]["entity"].split("-")[-1])
            and entity["index"] == entity_group_disagg[-1]["index"] + 1
        ):
            entity_group_disagg += [entity]
            # Group the entities at the last entity
            if is_last_idx:
                entity_groups += [group_sub_entities(entity_group_disagg)]
        # If the current entity is different from the previous entity, aggregate the disaggregated entity group
        else:
            entity_groups += [group_sub_entities(entity_group_disagg)]
            entity_group_disagg = [entity]
            # If it's the last entity, add it to the entity groups
            if is_last_idx:
                entity_groups += [group_sub_entities(entity_group_disagg)]

    return entity_groups

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

def init():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    global model
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), "outputs")
    with open(os.path.join(model_path, "model_config.json")) as f:
        model_config = json.load(f)
    
    model = BertNERPredictor(
        model_path=model_path,
        label_path=model_path,
        model_type=model_config["model_type"],
        do_lower_case=model_config.get("do_lower_case", "False") == "True",
        use_fast_tokenizer=model_config.get("use_fast_tokenizer", "True") == "True",
        device='cpu'
    )

def run(input_data):
    return model.predict(input_data)


# init()
# print(run("Steve went to Paris"))