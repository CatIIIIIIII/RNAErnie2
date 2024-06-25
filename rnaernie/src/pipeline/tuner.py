"""
This module builds up entry for running experiments.
Author: wangning(nwang227-c@my.cityu.edu.hk)
Date  : 2024/06/04 19:49
"""
from typing import Any, Dict, List, Optional

from transformers import TrainerCallback
from .pt import run_pt
from .infer import run_infer
from ..extras import LogCallback
from ..hparams import get_args


def run_exp(args: Optional[Dict[str, Any]] = None, callbacks: List["TrainerCallback"] = []) -> None:
    model_args, data_args, training_args, running_args, stage = get_args(
        args)
    callbacks.append(LogCallback(training_args.output_dir))

    if stage == "pt":
        run_pt(model_args, data_args,
               training_args, running_args, callbacks)
    elif stage == "infer":
        run_infer(model_args, data_args, training_args, running_args)
    else:
        raise ValueError("Unknown task.")
