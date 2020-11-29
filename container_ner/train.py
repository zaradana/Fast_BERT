#!/usr/bin/env python
# import mlflow
# import mlflow.pytorch
# from mlflow.utils.environment import _mlflow_conda_env

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

from azureml.core.run import Run
run = Run.get_context()

prefix = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(prefix, "data")
model_path = os.path.join(prefix, "models")
Path(model_path).mkdir(exist_ok=True)

def train(args):

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ],
    )
    logger = logging.getLogger()

    for k,v in args.items():
        run.log(k, v)

    model_name_path = args["model_name"]
 
    use_fast = args.get("do_lower_case", "False") == "True"
    tokenizer = AutoTokenizer.from_pretrained(model_name_path, use_fast=use_fast)

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
        use_fast_tokenizer=use_fast,
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
        warmup_steps=int(args["warmup_steps"]),
        grad_accumulation_steps=int(args["grad_accumulation_steps"]),
        multi_gpu=multi_gpu,
        logging_steps=int(args["logging_steps"]),
        save_steps=int(args["save_steps"]),
        adam_epsilon=float(args["adam_epsilon"])
    )

    learner.fit(int(args["epochs"]), float(args["lr"]))

    # Run validation
    evaluation_results = learner.validate()
    run.log('eval_f1', evaluation_results['eval_f1'])
    run.log('eval_recal', evaluation_results['eval_recall'])
    run.log('eval_precision', evaluation_results['eval_precision'])
    run.log('eval_loss', evaluation_results['eval_loss'])

    # save model and tokenizer artefacts
    output_path = Path("./outputs")
    learner.save_model(output_path)

    # save model config file
    with open(os.path.join(output_path, "model_config.json"), "w") as f:
        json.dump(args, f)

    # save label file
    with open(os.path.join(output_path, "labels.txt"), "w") as f:
        f.write("\n".join(databunch.labels))

 
if __name__ == "__main__":
    args = {}
    for s in sys.argv:
        if s == '--json_config':
            with open(os.path.abspath(sys.argv[sys.argv.index(s) + 1])) as json_file:
                args = json.load(json_file)
        elif s.startswith('--'):
            args[s.lstrip('--')] = sys.argv[sys.argv.index(s) + 1]

    train(args)