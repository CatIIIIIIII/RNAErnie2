import os
import shutil
import pickle
from tqdm import tqdm
import os.path as osp
from datetime import timedelta
from dataclasses import dataclass
from typing import Optional, List, Union, Dict, Any
import numpy as np

import torch
from transformers import Trainer
from transformers.trainer_utils import speed_metrics, has_length
from transformers.integrations.deepspeed import deepspeed_init
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs

from ...extras import get_logger
from ...hparams import InferenceArguments


logger = get_logger(__name__)
accelerotor = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(
    timeout=timedelta(seconds=18000))])


@dataclass
class InferDataCollatorWithPadding:
    """
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        ids = [f.pop("id") for f in features]
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch["id"] = torch.tensor(ids)
        return batch


class Inferencer(Trainer):
    r"""
    Inherits Trainer for custom optimizer.
    """

    def __init__(
        self, inference_args: InferenceArguments, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.inference_args = inference_args

    def predict_embeddings(self) -> List[torch.Tensor]:
        self.model.eval()
        inference_dataloader = self.get_eval_dataloader()
        dataset_embeddings = {}

        with torch.no_grad():
            with tqdm(self.num_examples(inference_dataloader), desc="Inferencing") as pbar:
                for batch in inference_dataloader:

                    ids = batch.pop("id").cpu().numpy()
                    inputs = {k: v.squeeze(0).to(self.accelerator.device)
                              for k, v in batch.items()}
                    outputs = self.model(**inputs, output_hidden_states=True)
                    embeddings = outputs.hidden_states[-1].half().detach().cpu()

                    # get the indices where input_ids is not self.tokenizer.pad_token
                    input_ids = inputs["input_ids"]
                    non_pad_indices = (
                        input_ids != self.tokenizer.pad_token_id).nonzero(as_tuple=True)
                    last_non_pad_indices = torch.zeros(
                        input_ids.size(0), dtype=torch.long)
                    for batch_idx in range(input_ids.size(0)):
                        batch_indices = non_pad_indices[1][non_pad_indices[0] == batch_idx]
                        if len(batch_indices) > 0:
                            last_non_pad_indices[batch_idx] = batch_indices[-1]

                    for i, embed in enumerate(embeddings):
                        embed = embed[1:last_non_pad_indices[i], :]
                        dataset_embeddings[ids[i]] = embed.numpy()

                    if accelerotor.is_main_process:
                        pbar.update(embeddings.shape[0] * self.args.world_size)

                    torch.cuda.empty_cache()

            with open(osp.join(self.args.output_dir, f"embeddings{accelerotor.process_index}.pickle"), "wb") as f:
                # sort the embeddings by id
                pickle.dump(dataset_embeddings, f)

            # if accelerotor.is_main_process:
                # merge the embeddings from different processes
            # embeddings = {}
            # for i in range(self.args.world_size):
            #     with open(osp.join(self.args.output_dir, f"embeddings{i}.pickle"), "rb") as f:
            #         embeddings.update(pickle.load(f))

                # with open(osp.join(self.args.output_dir, "rfam_crw_embeddings.pickle"), "wb") as f:
                #     pickle.dump(embeddings, f)
                # for i in range(self.args.world_size):
                #     os.remove(osp.join(self.args.output_dir,
                #               f"embeddings{i}.pickle"))
                # print(embeddings[0].shape)
        # return embeddings
