from transformers import Trainer

from ...extras import get_logger
from ...hparams import PretrainingArguments


logger = get_logger(__name__)


class PreTrainer(Trainer):
    r"""
    Inherits Trainer for custom optimizer.
    """

    def __init__(
        self, pretraining_args: PretrainingArguments, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.pretraining_args = pretraining_args
