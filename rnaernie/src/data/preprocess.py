from itertools import chain
from functools import partial
from typing import Any, Callable, Dict, List, Literal
import torch
from transformers.tokenization_utils import PreTrainedTokenizer

from ..extras.logging import get_logger
from ..hparams import DataArguments


logger = get_logger(__name__)


def preprocess_pretrain_dataset(
    examples: Dict[str, List[Any]],
    tokenizer: "PreTrainedTokenizer",
    data_args: "DataArguments"
) -> Dict[str, List[List[int]]]:
    result = tokenizer(examples["sequence"],
                       padding="do_not_pad" if data_args.do_group else 'longest',
                       max_length=data_args.max_seq_length,
                       truncation=True)
    return result


def preprocess_infer_dataset(
    examples: Dict[str, List[Any]],
    tokenizer: "PreTrainedTokenizer",
    data_args: "DataArguments"
) -> Dict[str, List[List[int]]]:
    result = tokenizer(examples["sequence"],
                       padding="do_not_pad" if data_args.do_group else 'longest',
                       max_length=data_args.max_seq_length,
                       truncation=True)
    return result


def get_preprocess(
    data_args: "DataArguments",
    stage: Literal["pt", "infer", "sft"],
    tokenizer: "PreTrainedTokenizer",
) -> Callable:
    if stage == "pt":
        preprocess_func = partial(
            preprocess_pretrain_dataset,
            tokenizer=tokenizer,
            data_args=data_args,
        )
    elif stage == "infer":
        preprocess_func = partial(
            preprocess_infer_dataset,
            tokenizer=tokenizer,
            data_args=data_args,
        )

    return preprocess_func


def print_pretrain_dataset_example(
        example: Dict[str, List[int]],
        tokenizer: PreTrainedTokenizer
) -> None:
    print("input_ids:\n{}".format(example["input_ids"]))
    print("inputs:\n{}".format(tokenizer.decode(
        example["input_ids"], skip_special_tokens=False)))


def print_infer_dataset_example(
        example: Dict[str, List[int]],
        tokenizer: PreTrainedTokenizer
) -> None:
    print("instance id: {}".format(example["id"]))
    print("input_ids:\n{}".format(example["input_ids"]))
    print("inputs:\n{}".format(tokenizer.decode(
        example["input_ids"], skip_special_tokens=False)))


def get_print_func(
    stage: Literal["pt", "sft"],
    tokenizer: "PreTrainedTokenizer",
) -> Callable:
    if stage == "pt":
        print_function = partial(
            print_pretrain_dataset_example, tokenizer=tokenizer)
    elif stage == "infer":
        print_function = partial(
            print_infer_dataset_example, tokenizer=tokenizer)

    return print_function


def group_pretrain_dataset(examples, data_args: DataArguments):
    # Concatenate all texts.
    concatenated_examples = {
        k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (
        total_length // data_args.max_seq_length) * data_args.max_seq_length
    # Split by chunks of max_len.
    result = {
        k: [t[i: i + data_args.max_seq_length]
            for i in range(0, total_length, data_args.max_seq_length)]
        for k, t in concatenated_examples.items()
    }
    return result
