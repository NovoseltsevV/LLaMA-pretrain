import torch
from torch.utils.data import DataLoader
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from datasets import Dataset
from torch import Tensor
from jaxtyping import Int
from typing import List, Union
from train_loop import TrainConfig
from model import ModelConfig
from functools import partial


def collate_fn(
    batch: List[dict], 
    max_seq_len: int,
    tokenizer: PreTrainedTokenizerBase,
    max_batch_size: int
) -> Union[Int[Tensor, "new_bs max_seq_len"], None]:
    texts_str = ' '.join(item['text'] for item in batch)
    tokens = tokenizer(
        texts_str, return_tensors='pt', add_special_tokens=False
    )['input_ids'].squeeze(0) # type: ignore
    num_tokens = len(tokens)
    new_bs = min(
        num_tokens // (max_seq_len - 2), max_batch_size
    )
    if new_bs == 0:
        return None
    tokens = tokens[:new_bs * (max_seq_len - 2)]
    bos_tensor = torch.full((new_bs, 1), tokenizer.bos_token_id, dtype=torch.long) # type: ignore
    eos_tensor = torch.full((new_bs, 1), tokenizer.eos_token_id, dtype=torch.long) # type: ignore
    return torch.cat(
        [bos_tensor, tokens.reshape(new_bs, -1), eos_tensor], dim=1 # type: ignore
    ).contiguous()

def create_dataloader(
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    interpolate: bool
) -> DataLoader:
    if interpolate:
        max_seq_len = model_cfg.new_context_len
    else:
        max_seq_len = model_cfg.context_len
    batch_size = train_cfg.bs_sentence
    loader = DataLoader(
        dataset=dataset, # type: ignore
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=partial(
            collate_fn,
            max_seq_len=max_seq_len, 
            tokenizer=tokenizer,
            max_batch_size=train_cfg.max_batch_size
        ),
        pin_memory=True
    )
    return loader
