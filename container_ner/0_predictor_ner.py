import os
import torch
import numpy as np
from fast_bert.data_ner import BertNERDataBunch
from torch import nn
from seqeval.metrics import f1_score, precision_score, recall_score
from typing import Dict, List, Optional, Tuple
from fast_bert.learner_util import Learner
from nltk import word_tokenize
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    EvalPrediction,
    HfArgumentParser,
    logging
)

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

        words = word_tokenize(text)
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
