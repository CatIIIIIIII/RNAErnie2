"""
This module builds up inference args for RNAErnie families.
Author: wangning(nwang227-c@my.cityu.edu.hk)
Date  : 2024/06/12 00:57
"""
from dataclasses import dataclass, field


@dataclass
class InferenceArguments:
    r"""
    Arguments pertaining to which techniques we are going to pre-train with.
    """
