"""
This module builds up parser for kinds of arguments.
Author: wangning(nwang227-c@my.cityu.edu.hk)
Date  : 2024/06/04 15:29
"""
import os
import sys
import logging
from typing import Any, Dict, Optional, Tuple

import transformers
from transformers import HfArgumentParser, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from ..hparams import StageArguments, ModelArguments, DataArguments, PretrainingArguments, InferenceArguments
from ..extras import get_logger, TRAINER_CONFIG, get_current_device

logger = get_logger(__name__)
_STAGE_ARGS = [StageArguments]
_STAGE_CLS = Tuple[StageArguments]
_PRETRAIN_ARGS = [ModelArguments, DataArguments,
                  TrainingArguments, PretrainingArguments]
_PRETRAIN_CLS = Tuple[ModelArguments, DataArguments,
                      TrainingArguments, PretrainingArguments]
_INFER_ARGS = [ModelArguments, DataArguments,
               TrainingArguments, InferenceArguments]
_INFER_CLS = Tuple[ModelArguments, DataArguments,
                   TrainingArguments, InferenceArguments]


def _set_transformers_logging(log_level: Optional[int] = logging.INFO) -> None:
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()


def _parse_args(parser: HfArgumentParser,
                args: Optional[Dict[str, Any]] = None) -> Tuple[Any]:
    if args is not None:
        return parser.parse_dict(args)

    if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        return parser.parse_yaml_file(os.path.abspath(sys.argv[1]), allow_extra_keys=True)

    (*parsed_args, unknown_args) = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    if unknown_args:
        print(parser.format_help())
        print("Got unknown args, potentially deprecated arguments: {}".format(unknown_args))
        raise ValueError(
            "Some specified arguments are not used by the HfArgumentParser: {}".format(unknown_args))

    return (*parsed_args,)


def _parse_stage_args(args: Optional[Dict[str, Any]] = None) -> _STAGE_CLS:
    parser = HfArgumentParser(_STAGE_ARGS)
    (args, ) = _parse_args(parser, args)

    return args.stage


def _parse_pretrain_args(args: Optional[Dict[str, Any]] = None) -> _PRETRAIN_CLS:
    parser = HfArgumentParser(_PRETRAIN_ARGS)
    return _parse_args(parser, args)


def _parse_infer_args(args: Optional[Dict[str, Any]] = None) -> _INFER_CLS:
    parser = HfArgumentParser(_INFER_ARGS)
    return _parse_args(parser, args)


def get_args(args: Optional[Dict[str, Any]] = None) -> Any:
    stage = _parse_stage_args(args)
    # remove the stage argument
    if stage == "pt":
        model_args, data_args, training_args, running_args = _parse_pretrain_args(
            args)
    elif stage == "infer":
        model_args, data_args, training_args, running_args = _parse_infer_args(
            args)
    else:
        raise ValueError("Unknown model pipeline stage.")
    # Setup logging
    if training_args.should_log:
        _set_transformers_logging()

    if (
        training_args.resume_from_checkpoint is None
        and training_args.do_train
        and os.path.isdir(training_args.output_dir)
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        files = os.listdir(training_args.output_dir)
        if last_checkpoint is None and len(files) > 0 and (len(files) != 1 or files[0] != TRAINER_CONFIG):
            raise ValueError(
                "Output directory already exists and is not empty. Please set `overwrite_output_dir`.")

        if last_checkpoint is not None:
            training_args.resume_from_checkpoint = last_checkpoint
            logger.info(
                "Resuming training from {}. Change `output_dir` or use `overwrite_output_dir` to avoid.".format(
                    training_args.resume_from_checkpoint
                )
            )

    model_args.device_map = {"": get_current_device()}
    model_args.model_max_length = data_args.max_seq_length

    # Log on each process the small summary
    logger.info(
        "Process rank: {}, device: {}, n_gpu: {}, distributed training: {}".format(
            training_args.local_rank,
            training_args.device,
            training_args.n_gpu,
            training_args.parallel_mode.value == "distributed",
        )
    )
    training_args.ddp_find_unused_parameters = False
    transformers.set_seed(training_args.seed)

    return model_args, data_args, training_args, running_args, stage
