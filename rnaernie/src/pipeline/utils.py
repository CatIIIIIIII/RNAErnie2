
from transformers import Trainer
from transformers import TrainingArguments
from ..extras.logging import get_logger
from ..hparams import DataArguments, ModelArguments, PretrainingArguments


logger = get_logger(__name__)
_RUN_ARGS = PretrainingArguments


def create_modelcard_and_push(
    trainer: Trainer,
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: TrainingArguments,
    running_args: _RUN_ARGS,
) -> None:
    kwargs = {
        "tasks": "text-generation",
        "finetuned_from": model_args.model_name_or_path,
        "tags": ["llama-factory", running_args.finetuning_type],
    }
    if data_args.dataset is not None:
        kwargs["dataset"] = [dataset.strip()
                             for dataset in data_args.dataset.split(",")]

    if model_args.use_unsloth:
        kwargs["tags"] = kwargs["tags"] + ["unsloth"]

    if not training_args.do_train:
        pass
    elif training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        # prevent from connecting to hub
        trainer.create_model_card(license="other", **kwargs)
