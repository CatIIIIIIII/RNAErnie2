# Inspired by: https://github.com/huggingface/transformers/blob/v4.34.1/examples/pytorch/language-modeling/run_clm.py

import math
from typing import List, Optional

from transformers import TrainingArguments, TrainerCallback

from .trainer import PreTrainer
from ...extras import plot_loss
from ...data import get_dataset, load_motif, PretrainDataCollatorwithMotif
from ...models import load_model, load_tokenizer
from ...hparams import DataArguments, PretrainingArguments, ModelArguments


def run_pt(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "TrainingArguments",
    pretraining_args: "PretrainingArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):

    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    train_dataset, eval_dataset = get_dataset(
        model_args, data_args, training_args, stage="pt", **tokenizer_module)
    # load motifs
    motif_tree_dict = load_motif(
        motif_dir=data_args.motif_dir, tokenizer=tokenizer)
    data_collator = PretrainDataCollatorwithMotif(
        tokenizer=tokenizer, motif_tree=motif_tree_dict)

    model = load_model(model_args, training_args.do_train)
    # Initialize our Trainer
    trainer = PreTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        pretraining_args=pretraining_args,
        data_collator=data_collator,
        callbacks=callbacks,
        **tokenizer_module,
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train(
            resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and pretraining_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss"])

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval")
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")

        metrics["perplexity"] = perplexity
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


# def run_pt(
#     model_args: "ModelArguments",
#     data_args: "DataArguments",
#     training_args: "TrainingArguments",
#     pretraining_args: "PretrainingArguments",
#     callbacks: Optional[List["TrainerCallback"]] = None,
# ):
#     tokenizer_module = load_tokenizer(model_args)
#     tokenizer = tokenizer_module["tokenizer"]
#     dataset = get_dataset(model_args, data_args,
#                           training_args, stage="pt", **tokenizer_module)
#     # load motifs
#     motif_tree_dict = load_motif(
#         motif_dir=data_args.motif_dir, tokenizer=tokenizer)
#     data_collator = PretrainDataCollatorwithMotif(
#         tokenizer=tokenizer, motif_tree=motif_tree_dict)

#     model = load_model(tokenizer, model_args,
#                        pretraining_args, training_args.do_train)
#     # Initialize our Trainer
#     trainer = PreTrainer(
#         model=model,
#         args=training_args,
#         pretraining_args=pretraining_args,
#         data_collator=data_collator,
#         callbacks=callbacks,
#         **tokenizer_module,
#         **split_dataset(dataset, data_args, training_args),
#     )

#     with tqdm(159_000) as pbar:
#         for batch in trainer.get_train_dataloader():
#             pbar.update(1)
