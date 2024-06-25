"""
This module builds up model stage arguments.
Author: wangning(nwang227-c@my.cityu.edu.hk)
Date  : 2024/06/04 19:14
"""
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class StageArguments:
    """
    Arguments to indicate the stage of model.
    """

    stage: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model pipeline stage."
            )
        },
    )
