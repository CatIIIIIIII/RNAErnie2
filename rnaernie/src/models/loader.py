from typing import Any, Dict, TypedDict

from transformers import AutoConfig, AutoModelForMaskedLM

from ..extras.logging import get_logger
from ..extras.misc import count_parameters
from .patcher import patch_tokenizer

from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizer
from .rnaernie.tokenization_rnaernie import RNAErnieTokenizer
from ..hparams import ModelArguments

logger = get_logger(__name__)


class TokenizerModule(TypedDict):
    tokenizer: "PreTrainedTokenizer"


def _get_init_kwargs(model_args: ModelArguments) -> Dict[str, Any]:
    r"""
    Gets arguments to load config/tokenizer/model.

    Note: including inplace operation of model_args.
    """
    return {
        "trust_remote_code": True,
        "cache_dir": model_args.cache_dir,
        # "revision": model_args.model_revision,
        # "token": model_args.hf_hub_token,
    }


def load_tokenizer(model_args: ModelArguments) -> TokenizerModule:
    r"""
    Loads pretrained tokenizer.

    Note: including inplace operation of model_args.
    """
    init_kwargs = _get_init_kwargs(model_args)

    try:
        tokenizer = RNAErnieTokenizer.from_pretrained(
            model_args.model_name_or_path,
            use_fast=model_args.use_fast_tokenizer,
            padding_side="right",
            **init_kwargs,
        )
    except ValueError:
        tokenizer = RNAErnieTokenizer.from_pretrained(
            model_args.model_name_or_path,
            use_fast=True,
            padding_side="right",
            **init_kwargs,
        )

    patch_tokenizer(tokenizer)
    return {"tokenizer": tokenizer}


def load_config(model_args: ModelArguments) -> PretrainedConfig:
    r"""
    Loads model config.
    """
    init_kwargs = _get_init_kwargs(model_args)
    return AutoConfig.from_pretrained(model_args.model_name_or_path, **init_kwargs)


def load_model(
    model_args: ModelArguments,
    is_trainable: bool = False,
) -> PreTrainedModel:
    r"""
    Loads pretrained model.
    """
    init_kwargs = _get_init_kwargs(model_args)
    config = load_config(model_args)

    init_kwargs["config"] = config
    init_kwargs["pretrained_model_name_or_path"] = model_args.model_name_or_path
    model = AutoModelForMaskedLM.from_pretrained(**init_kwargs)

    if not is_trainable:
        model.requires_grad_(False)
        model.eval()
    else:
        model.train()

    trainable_params, all_param = count_parameters(model)
    if is_trainable:
        param_stats = "trainable params: {:d} || all params: {:d} || trainable%: {:.4f}".format(
            trainable_params, all_param, 100 * trainable_params / all_param
        )
    else:
        param_stats = "all params: {:d}".format(all_param)
    logger.info(param_stats)

    return model
