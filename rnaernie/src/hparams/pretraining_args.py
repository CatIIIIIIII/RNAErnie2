"""
This module builds up args for pre-training.
Author: wangning(nwang227-c@my.cityu.edu.hk)
Date  : 2024/06/04 13:34
"""
from dataclasses import dataclass, field


@dataclass
class PretrainingArguments:
    r"""
    Arguments pertaining to which techniques we are going to pre-train with.
    """
    plot_loss: bool = field(
        default=False,
        metadata={"help": "Whether or not to save the training loss curves."},
    )
