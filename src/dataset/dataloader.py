from torch.utils.data import DataLoader
from torch.utils.data.distributed import (
    DistributedSampler
)

from dataset.base_dataset import (
    BaseDataset
)

import torch.distributed as dist

import torch

def custom_collate(batch):

    out = {}

    for k in batch[0].keys():

        values = [d[k] for d in batch]

        # images → KEEP LIST (NO STACK)
        if k == "image":
            out[k] = values

        # tensors → stack
        elif torch.is_tensor(values[0]):
            out[k] = torch.stack(values, dim=0)

        # strings / paths / labels
        else:
            out[k] = values

    return out

def build_dataloader(
    csv_path,
    dataset_config,
    training_config,
    processors,
    modalities,
    tokenizer=None,
    split="train"
):

    dataset = BaseDataset(

        csv_path=csv_path,

        config=dataset_config,

        processors=processors,

        modalities=modalities,

        tokenizer=tokenizer,

        split=split
    )

    sampler = None

    # DDP
    if (
        dist.is_available()
        and dist.is_initialized()
        and split == "train"
    ):
    
        sampler = DistributedSampler(
            dataset,
            shuffle=True
        )

    dataloader = DataLoader(

        dataset,

        batch_size=training_config[
            "batch_size"
        ],

        collate_fn=custom_collate,

        shuffle=(
            sampler is None
            and split == "train"
        ),

        sampler=sampler,

        num_workers=training_config[
            "num_workers"
        ],

        pin_memory=True
    )

    return dataloader