#!/usr/bin/env python
import mlflow
import mlflow.pytorch
from mlflow.utils.environment import _mlflow_conda_env

import os
import json
import pickle
import sys
import traceback
import pandas as pd
import datetime
from pathlib import Path
import logging
import inspect
import torch
from transformers import AutoTokenizer

from fast_bert.data_ner import BertNERDataBunch
from fast_bert.learner_ner import BertNERLearner


channel_name = "training"

prefix = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(prefix, "data")
model_path = os.path.join(prefix, "model")
Path(model_path).mkdir(exist_ok=True)

def train(args):
    with mlflow.start_run():
        params = {}
        for k,v in args:
            params[k] = v
        mlflow.log_params(params)
    
        logger = logging.getLogger()

        model_name_path = args["model_name"]

        tokenizer = AutoTokenizer.from_pretrained(model_name_path, use_fast=True)

        device = torch.device("cuda")
        if torch.cuda.device_count() > 1:
            multi_gpu = True
        else:
            multi_gpu = False

        logger.info("Number of GPUs: {}".format(torch.cuda.device_count()))

        databunch = BertNERDataBunch(
            data_dir=data_path,
            tokenizer=tokenizer,
            batch_size_per_gpu=int(args["train_batch_size"]),
            max_seq_length=int(args["max_seq_length"]),
            multi_gpu=multi_gpu,
            backend="nccl",
            model_type=args["model_type"],
            logger=logger,
            clear_cache=True,
            no_cache=False,
            use_fast_tokenizer=args["use_fast_tokenizer"],
            custom_sampler=None
        )

        logger.info("databunch labels: {}".format(len(databunch.labels)))

        # Initialise the learner
        learner = BertNERLearner.from_pretrained_model(
            databunch,
            model_name_path,
            output_dir=Path(model_path),
            device=device,
            logger=logger,
            finetuned_wgts_path=None,
            is_fp16=args["fp16"],
            fp16_opt_level=args["fp16_opt_level"],
            warmup_steps=int(args["warmup_steps"]),
            grad_accumulation_steps=int(args["grad_accumulation_steps"]),
            multi_gpu=multi_gpu,
            logging_steps=int(args["logging_steps"]),
            save_steps=int(args["save_steps"]),
            adam_epsilon=int(args["adam_epsilon"])
        )

        learner.fit(int(args["epochs"]), float(args["lr"]))

        # Run validation
        logger.info(learner.validate())

        # save model and tokenizer artefacts
        # learner.save_model()

        # save model config file
        with open(os.path.join(model_path, "model_config.json"), "w") as f:
            json.dump(args, f)

        # save label file
        with open(os.path.join(model_path, "labels.txt"), "w") as f:
            f.write("\n".join(databunch.labels))

        model_env = _mlflow_conda_env()
        mlflow.pytorch.log_model(learner.model, "model", conda_env=model_env)

if __name__ == "__main__":
    json_config_arg = '--json_config'
    if json_config_arg in sys.argv:
        json_conf_index = sys.argv.index(json_config_arg)
        with open(os.path.abspath(sys.argv[json_conf_index + 1])) as json_file:
            args = json.load(json_file)

    train(args)