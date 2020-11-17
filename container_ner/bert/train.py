#!/usr/bin/env python
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
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import torch
from transformers import AutoTokenizer

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
grandparentdir = os.path.dirname(parentdir)
sys.path.insert(0, grandparentdir)

from fast_bert.data_ner import BertNERDataBunch
from fast_bert.learner_ner import BertNERLearner

run_start_time = datetime.datetime.today().strftime("%Y-%m-%d_%H-%M-%S")

channel_name = "training"

prefix = "./opt/ml/"
input_path = prefix + "input/data"  # opt/ml/input/data
code_path = prefix + "code"  # opt/ml/code
pretrained_model_path = (
    code_path + "/pretrained_models"
)  # opt/ml/code/pretrained_models

finetuned_path = input_path + "/{}/finetuned".format(
    channel_name
)  # opt/ml/input/data/training/finetuned

output_path = os.path.join(prefix, "output")  # opt/ml/output
model_path = os.path.join(prefix, "model")  # opt/ml/model

training_config_path = os.path.join(
    input_path, "{}/config".format(channel_name)
)  # opt/ml/input/data/training/config

hyperparam_path = os.path.join(
    prefix, "input/config/hyperparameters.json"
)  # opt/ml/input/config/hyperparameters.json
config_path = os.path.join(
    training_config_path, "training_config.json"
)  # opt/ml/input/data/training/config/training_config.json


# This algorithm has a single channel of input data called 'training'. Since we run in
# File mode, the input files are copied to the directory specified here.

training_path = os.path.join(input_path, channel_name)  # opt/ml/input/data/training


def searching_all_files(directory: Path):
    file_list = []  # A list for storing files existing in directories

    for x in directory.iterdir():
        if x.is_file():
            file_list.append(str(x))
        else:
            file_list.append(searching_all_files(x))

    return file_list


# The function to execute the training.
def train():

    print("Starting the training.")

    DATA_PATH = Path(training_path)

    try:
        print(config_path)
        with open(config_path, "r") as f:
            training_config = json.load(f)
            print(training_config)

        with open(hyperparam_path, "r") as tc:
            hyperparameters = json.load(tc)
            print(hyperparameters)

        # convert string bools to booleans
        training_config["fp16"] = training_config["fp16"] == "True"
        training_config["use_fast_tokenizer"] = (
            training_config.get("use_fast_tokenizer", "True") == "True"
        )
        training_config["jsonl_file"] = training_config.get("jsonl_file", "data.jsonl")

        training_config["random_state"] = (
            int(training_config.get("random_state"))
            if training_config.get("random_state")
            else None
        )

        training_config["train_size"] = float(training_config.get("train_size", 0.8))

        # Logger
        # logfile = str(LOG_PATH/'log-{}-{}.txt'.format(run_start_time, training_config["run_text"]))
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            handlers=[
                # logging.FileHandler(logfile),
                logging.StreamHandler(sys.stdout)
            ],
        )

        logger = logging.getLogger()

        # Define pretrained model path
        PRETRAINED_PATH = Path(pretrained_model_path) / training_config["model_name"]
        if PRETRAINED_PATH.is_dir():
            logger.info("model path used {}".format(PRETRAINED_PATH))
            model_name_path = str(PRETRAINED_PATH)
        else:
            model_name_path = training_config["model_name"]
            logger.info(
                "model {} is not preloaded. Will try to download.".format(
                    model_name_path
                )
            )

        finetuned_model_name = training_config.get("finetuned_model", None)
        if finetuned_model_name is not None:
            finetuned_model = os.path.join(finetuned_path, finetuned_model_name)
            logger.info("finetuned model loaded from {}".format(finetuned_model))
        else:
            logger.info(
                "finetuned model not available - loading standard pretrained model"
            )
            finetuned_model = None

        # use auto-tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name_path, use_fast=True)

        device = torch.device("cuda")
        multi_gpu = False
        # if torch.cuda.device_count() > 1:
        #     multi_gpu = True
        # else:
        #     multi_gpu = False

        logger.info("Number of GPUs: {}".format(torch.cuda.device_count()))

        databunch = BertNERDataBunch(
            data_dir=input_path,
            tokenizer=tokenizer,
            batch_size_per_gpu=int(hyperparameters["train_batch_size"]),
            max_seq_length=int(hyperparameters["max_seq_length"]),
            multi_gpu=multi_gpu,
            backend="nccl",
            model_type=training_config["model_type"],
            logger=logger,
            clear_cache=True,
            no_cache=False,
            use_fast_tokenizer=training_config["use_fast_tokenizer"],
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
            finetuned_wgts_path=finetuned_model,
            is_fp16=training_config["fp16"],
            fp16_opt_level=training_config["fp16_opt_level"],
            warmup_steps=int(hyperparameters["warmup_steps"]),
            grad_accumulation_steps=int(training_config["grad_accumulation_steps"]),
            multi_gpu=multi_gpu,
            logging_steps=int(training_config["logging_steps"]),
            save_steps=int(training_config.get("save_steps", 0)),
            adam_epsilon=int(hyperparameters["adam_epsilon"])
        )

        learner.fit(int(hyperparameters["epochs"]), float(hyperparameters["lr"]))

        # Run validation
        logger.info(learner.validate())

        # save model and tokenizer artefacts
        learner.save_model()

        # save model config file
        with open(os.path.join(model_path, "model_config.json"), "w") as f:
            json.dump(training_config, f)

        # save label file
        with open(os.path.join(model_path, "labels.txt"), "w") as f:
            f.write("\n".join(databunch.labels))

    except Exception as e:
        # Write out an error file. This will be returned as the failureReason in the
        # DescribeTrainingJob result.
        trc = traceback.format_exc()
        with open(os.path.join(output_path, "failure"), "w") as s:
            s.write("Exception during training: " + str(e) + "\n" + trc)
        # Printing this causes the exception to be in the training job logs, as well.
        print("Exception during training: " + str(e) + "\n" + trc, file=sys.stderr)
        # A non-zero exit code causes the training job to be marked as Failed.
        sys.exit(255)


if __name__ == "__main__":
    train()

    # A zero exit code causes the job to be marked a Succeeded.
    sys.exit(0)
