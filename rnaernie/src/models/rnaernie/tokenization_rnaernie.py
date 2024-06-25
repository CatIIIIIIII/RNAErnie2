"""
This module builds up tokenization for RNAErnie.

Author: wangning(nwang227-c@my.cityu.edu.hk)
Date  : 2024/06/04 13:18
"""
from transformers.utils import logging
from ..tokenization_utils import RNATokenizer

logger = logging.get_logger(__name__)


class RNAErnieTokenizer(RNATokenizer):
    """
    Constructs an RNAErnie tokenizer.

    Args:
        vocab_file (str): Path to the vocabulary file.
        do_upper_case (bool, optional): Whether to convert input to uppercase.
            Defaults to True.
        replace_T_with_U (bool, optional): Whether to replace T with U.
            Defaults to True.
    """

    def __init__(
        self,
        vocab_file,
        do_upper_case: bool = True,
        replace_T_with_U: bool = True,
        **kwargs,
    ):
        super().__init__(
            vocab_file=vocab_file,
            do_upper_case=do_upper_case,
            replace_T_with_U=replace_T_with_U,
            **kwargs,
        )
