# Inspired by: https://github.com/huggingface/transformers/blob/v4.34.1/examples/pytorch/language-modeling/run_clm.py

from typing import List, Optional
from transformers import TrainingArguments
import os.path as osp
import pickle
import os

from .inferencer import Inferencer, InferDataCollatorWithPadding
from ...extras import plot_loss
from ...data import get_dataset
from ...models import load_model, load_tokenizer
from ...hparams import DataArguments, ModelArguments, InferenceArguments


def run_infer(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "TrainingArguments",
    inference_args: "InferenceArguments",
):
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    _, infer_dataset = get_dataset(
        model_args, data_args, training_args, stage="infer", **tokenizer_module)
    data_collator = InferDataCollatorWithPadding(
        tokenizer=tokenizer, padding=True)

    assert not training_args.do_train, "Inference requires do_train=False"
    model = load_model(model_args, training_args.do_train)
    # Initialize our Trainer
    inferencer = Inferencer(
        model=model,
        args=training_args,
        inference_args=inference_args,
        eval_dataset=infer_dataset,
        data_collator=data_collator,
        **tokenizer_module,
    )

    inferencer.predict_embeddings()
    # embeddings = {}

    # for i in range(training_args.world_size):
    #     with open(osp.join(training_args.output_dir, f"embeddings{i}.pickle"), "rb") as f:
    #         embeddings.update(pickle.load(f))

    #     with open(osp.join(training_args.output_dir, "rfam_crw_embeddings.pickle"), "wb") as f:
    #         pickle.dump(embeddings, f)
    #     for i in range(training_args.world_size):
    #         os.remove(osp.join(training_args.output_dir,
    #                            f"embeddings{i}.pickle"))
