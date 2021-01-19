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
import math
from dataclasses import dataclass, field
from typing import Optional
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch

from transformers import (
    CONFIG_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    LineByLineTextDataset,
    PreTrainedTokenizer,
    TextDataset,
    Trainer,
    TrainingArguments,
    set_seed,
)

run_start_time = datetime.datetime.today().strftime("%Y-%m-%d_%H-%M-%S")

channel_name = "training"

prefix = "./opt/ml/"
input_path = prefix + "input/data"  # opt/ml/input/data
code_path = prefix + "code"  # opt/ml/code

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


training_path = input_path #os.path.join(input_path, channel_name)  # opt/ml/input/data/training

MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "If training from scratch, pass a model type from the list: "
            + ", ".join(MODEL_TYPES)
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from s3"
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_data_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    eval_data_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."
        },
    )
    line_by_line: bool = field(
        default=True,
        metadata={
            "help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."
        },
    )

    mlm: bool = field(
        default=True,
        metadata={
            "help": "Train with masked-language modeling loss instead of language modeling."
        },
    )
    mlm_probability: float = field(
        default=0.15,
        metadata={"help": "Ratio of tokens to mask for masked language modeling loss"},
    )
    plm_probability: float = field(
        default=1 / 6,
        metadata={
            "help": "Ratio of length of a span of masked tokens to surrounding context length for permutation language modeling."
        },
    )
    max_span_length: int = field(
        default=5,
        metadata={
            "help": "Maximum length of a span of masked tokens for permutation language modeling."
        },
    )

    block_size: int = field(
        default=-1,
        metadata={
            "help": "Optional input sequence length after tokenization."
            "The training dataset will be truncated in block of this size for training."
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )


def get_dataset(
    args: DataTrainingArguments, tokenizer: PreTrainedTokenizer, evaluate=False
):
    file_path = args.eval_data_file if evaluate else args.train_data_file
    if args.line_by_line:
        return LineByLineTextDataset(
            tokenizer=tokenizer, file_path=file_path, block_size=args.block_size
        )
    else:
        return TextDataset(
            tokenizer=tokenizer,
            file_path=file_path,
            block_size=args.block_size,
            overwrite_cache=args.overwrite_cache,
        )


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
        training_config["train_file"] = training_config.get("train_file", "train.csv")
        training_config["val_file"] = training_config.get("val_file", "val.csv")
        training_config["fp16"] = training_config["fp16"] == "True"
        training_config["line_by_line"] = training_config["line_by_line"] == "True"
        training_config["do_lower_case"] = training_config["do_lower_case"] == "True"
        training_config["tokenizer_name"] = training_config["tokenizer_name"] 
        training_config["use_fast_tokenizer"] = (
            training_config.get("use_fast_tokenizer", "True") == "True"
        )
        training_config["mlm"] = training_config["mlm"] == "True"
        training_config["mlm_probability"] = float(
            training_config.get("mlm_probability", 0.15)
        )
        training_config["block_size"] = int(training_config.get("block_size", -1))

        training_config["random_state"] = (
            int(training_config.get("random_state"))
            if training_config.get("random_state")
            else None
        )

        # training_config["train_size"] = float(training_config.get("train_size", 0.8))

        data_args = DataTrainingArguments(
            train_data_file=str(DATA_PATH / training_config["train_file"]),
            eval_data_file=str(DATA_PATH / training_config["val_file"]),
            line_by_line=training_config["line_by_line"],
            mlm=training_config["mlm"],
            mlm_probability=training_config["mlm_probability"],
            block_size=training_config["block_size"],
        )

        training_args = TrainingArguments(
            output_dir=model_path,
            overwrite_output_dir=True,
            do_train=True,
            do_eval=True,
            evaluate_during_training=True,
            per_device_train_batch_size=int(hyperparameters["train_batch_size"]),
            per_device_eval_batch_size=int(hyperparameters["train_batch_size"]) * 2,
            gradient_accumulation_steps=int(training_config["grad_accumulation_steps"]),
            warmup_steps=int(hyperparameters["warmup_steps"]),
            logging_steps=int(training_config["logging_steps"]),
            fp16=training_config["fp16"],
            fp16_opt_level=training_config["fp16_opt_level"],
            seed=training_config["random_state"],
            num_train_epochs=int(hyperparameters["epochs"]),
            learning_rate=float(hyperparameters["lr"]),
            adam_epsilon=int(hyperparameters["adam_epsilon"]),
            save_steps=0,
        )

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

        set_seed(training_args.seed)

        # use auto-tokenizer
        # tokenizer = AutoTokenizer.from_pretrained(
        #     training_config["model_name"],
        #     use_fast=training_config["use_fast_tokenizer"],
        # )
        tokenizer = AutoTokenizer.from_pretrained(training_config["tokenizer_name"] 
                if training_config["tokenizer_name"] else training_config["model_name_path"], 
                use_fast=training_config["use_fast_tokenizer"], do_lower_case=training_config["do_lower_case"],
                cache_dir=str(output_path))

        config = AutoConfig.from_pretrained(training_config["model_name"])

        model = AutoModelWithLMHead.from_pretrained(
            training_config["model_name"], config=config
        )
        model.resize_token_embeddings(len(tokenizer))

        if (
            config.model_type in ["bert", "roberta", "distilbert", "camembert"]
            and not data_args.mlm
        ):
            raise ValueError(
                "BERT and RoBERTa-like models do not have LM heads but masked LM heads. They must be run using the"
                "--mlm flag (masked language modeling)."
            )

        if data_args.block_size <= 0:
            data_args.block_size = tokenizer.max_len
            # Our input block size will be the max possible for the model
        else:
            data_args.block_size = min(data_args.block_size, tokenizer.max_len)

        # Get datasets

        train_dataset = get_dataset(data_args, tokenizer=tokenizer)
        eval_dataset = get_dataset(data_args, tokenizer=tokenizer, evaluate=True)

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=data_args.mlm,
            mlm_probability=data_args.mlm_probability,
        )

        # Initialize our Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            prediction_loss_only=True,
        )

        # Run pre-validation
        if training_args.do_eval:
            logger.info("*** Evaluate before training ***")
            logger.info(validate(trainer, logger))

        trainer.train()

        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

        # Run validation
        if training_args.do_eval:
            logger.info("*** Evaluate ***")
            logger.info(validate(trainer, logger))

        # save model config file
        with open(os.path.join(model_path, "model_config.json"), "w") as f:
            json.dump(training_config, f)

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


def validate(trainer: Trainer, logger):
    results = {}
    eval_output = trainer.evaluate()

    perplexity = math.exp(eval_output["eval_loss"])
    result = {"perplexity": perplexity}

    results.update(result)

    return results


if __name__ == "__main__":
    train()

    # A zero exit code causes the job to be marked a Succeeded.
    sys.exit(0)
