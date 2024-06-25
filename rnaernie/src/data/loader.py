from ..hparams import DataArguments, ModelArguments
import sys
from pathlib import Path
from ahocorapy.keywordtree import KeywordTree

from typing import Literal, Union
from datasets import load_dataset, load_from_disk, Dataset, IterableDataset, Features, Sequence, Value
from functools import partial
from datetime import timedelta
from ..extras import get_logger, has_tokenized_data, has_file
from .preprocess import get_preprocess, get_print_func, group_pretrain_dataset

from transformers import TrainingArguments
from transformers.tokenization_utils import PreTrainedTokenizer
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs

accelerotor = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(
    timeout=timedelta(seconds=18000))])


logger = get_logger(__name__)


def get_dataset(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "TrainingArguments",
    stage: Literal["pt", "infer", "sft"],
    tokenizer: "PreTrainedTokenizer",
) -> Union["Dataset", "IterableDataset"]:

    if data_args.tokenized_path is not None:
        if has_tokenized_data(data_args.tokenized_path):
            logger.warning(
                "Loading dataset from disk will ignore other data arguments.")
            tokenized_datasets = load_from_disk(data_args.tokenized_path)
            logger.info("Loaded tokenized dataset from {}.".format(
                data_args.tokenized_path))
        else:
            # with training_args.main_process_first(desc="load dataset"):
            if data_args.dataset_name is not None:
                # Downloading and loading a dataset from the hub.
                raw_datasets = load_dataset(
                    data_args.dataset_name,
                    data_args.dataset_config_name,
                    cache_dir=model_args.cache_dir,
                    token=model_args.token)
                if training_args.do_train and "validation" not in raw_datasets.keys():
                    raw_datasets["validation"] = load_dataset(
                        data_args.dataset_name,
                        data_args.dataset_config_name,
                        split=f"train[:{data_args.validation_split_percentage}%]",
                        cache_dir=model_args.cache_dir,
                        token=model_args.token)
                    raw_datasets["train"] = load_dataset(
                        data_args.dataset_name,
                        data_args.dataset_config_name,
                        split=f"train[{data_args.validation_split_percentage}%:]",
                        cache_dir=model_args.cache_dir,
                        token=model_args.token)
                else:
                    raw_datasets["validation"] = load_dataset(
                        data_args.dataset_name,
                        data_args.dataset_config_name,
                        cache_dir=model_args.cache_dir,
                        split=f"train[:{data_args.validation_split_percentage}%]",
                        token=model_args.token)

            else:
                data_files = {}
                if data_args.train_file is not None:
                    data_files["train"] = data_args.train_file
                    extension = data_args.train_file.split(".")[-1]
                if data_args.validation_file is not None:
                    data_files["validation"] = data_args.validation_file
                    extension = data_args.validation_file.split(".")[-1]
                if extension == "txt":
                    extension = "text"
                raw_datasets = load_dataset(
                    extension,
                    data_files=data_files,
                    cache_dir=model_args.cache_dir,
                    token=model_args.token)

                if "validation" not in raw_datasets.keys():
                    raw_datasets["validation"] = load_dataset(
                        extension,
                        data_files=data_files,
                        split=f"train[:{data_args.validation_split_percentage}%]",
                        cache_dir=model_args.cache_dir,
                        token=model_args.token,
                    )
                    raw_datasets["train"] = load_dataset(
                        extension,
                        data_files=data_files,
                        split=f"train[{data_args.validation_split_percentage}%:]",
                        cache_dir=model_args.cache_dir,
                        token=model_args.token,
                    )

            preprocess_func = get_preprocess(data_args, stage, tokenizer)
            # do tokenization
            if training_args.do_train:
                column_names = list(raw_datasets["train"].features)
            else:
                column_names = list(raw_datasets["validation"].features)

            if accelerotor.is_local_main_process:
                if "id" not in column_names:
                    features = Features({
                        "input_ids": Sequence(Value("int8")),
                        "attention_mask": Sequence(Value("bool")),
                    })
                else:
                    assert not training_args.do_train, "id column is only required for prediction dataset"
                    features = Features({
                        "id": Value("int64"),
                        "input_ids": Sequence(Value("int8")),
                        "attention_mask": Sequence(Value("bool")),
                    })
                # map feature function
                column_names.remove('id')
                tokenized_datasets = raw_datasets.map(
                    preprocess_func,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                    features=features,
                    desc="Running tokenizer on every text in dataset",
                )

                # do group
                if stage == "pt" and data_args.do_group:
                    group_pretrain_func = partial(
                        group_pretrain_dataset, data_args=data_args)
                    tokenized_datasets = tokenized_datasets.map(
                        group_pretrain_func,
                        batched=True,
                        num_proc=data_args.preprocessing_num_workers,
                        load_from_cache_file=not data_args.overwrite_cache,
                        desc=f"Grouping texts in chunks of {data_args.max_seq_length}",
                    )

                if data_args.tokenized_path is not None:
                    tokenized_datasets.save_to_disk(data_args.tokenized_path)
                    logger.info("Tokenized dataset saved at {}.".format(
                        data_args.tokenized_path))
                    logger.info("Please restart the training with `tokenized_path: {}`.".format(
                        data_args.tokenized_path))

                    sys.exit(0)

                # save tokenized dataset
                accelerotor.wait_for_everyone()

            else:
                accelerotor.wait_for_everyone()
                if data_args.tokenized_path is not None:
                    tokenized_datasets = load_from_disk(
                        data_args.tokenized_path)

    train_dataset = None
    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = tokenized_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(
                len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(
                range(max_train_samples))

    eval_dataset = None
    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = tokenized_datasets["validation"]

    if training_args.should_log:
        print_function = get_print_func(stage, tokenizer)
        if stage == "pt":
            data_example = next(iter(train_dataset)) if train_dataset \
                else next(iter(eval_dataset))
            print_function(data_example)
    return train_dataset, eval_dataset


def load_motif(motif_dir, tokenizer):
    """load motifs from file

    Args:
        motif_dir (str): motif data root directory
        tokenizer (tokenizer_nuc.NUCTokenizer): nucleotide tokenizer

    Returns:
        dict: {file_name: [int]}
    """
    motif_dict = {}
    motif_name = [str(file) for file in Path(motif_dir).rglob('*.txt')]
    for name in motif_name:
        with open(name, 'r') as f:
            motifs = f.readlines()

        motif_tokens = []
        for m in motifs:
            input_ids = tokenizer(
                m.replace("\n", ""), return_token_type_ids=False)["input_ids"]
            input_ids = input_ids[1:-1]
            motif_tokens.append(input_ids)
            print(motif_tokens)
        motif_dict[name.split('/')[-1].split('.')[0]] = motif_tokens

    motif_tree_dict = {}
    motif_tree = KeywordTree()
    for k, v in motif_dict.items():
        if k != "Statistics":
            for m in v:
                motif_tree.add(m)
    motif_tree.finalize()
    motif_tree_dict["DataBases"] = motif_tree

    motif_tree = KeywordTree()
    for k, v in motif_dict.items():
        if k == "Statistics":
            for m in v:
                motif_tree.add(m)
    motif_tree.finalize()
    motif_tree_dict["Statistics"] = motif_tree

    return motif_tree_dict
