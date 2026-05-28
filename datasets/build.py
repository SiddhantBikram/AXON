import os
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data.dataloader import default_collate
from collections.abc import Mapping, Sequence
from functools import partial
from sklearn.model_selection import train_test_split

class CSVNpyDataset(Dataset):
    """
    A dataset that:
     - Reads a CSV with columns [File, Function]
     - Loads .npy arrays from the 'File' column
     - Uses the 'Function' column as labels
    """
    def __init__(self, df, label = 'ACTION1_ID',transform=None):
        """
        Args:
            csv_path (str): Path to the CSV file.
            transform (callable, optional): Optional transform to apply to each loaded npy array.
        """
        super().__init__()
        self.df = df
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = os.path.join('', row["sample_name"])
        label = row[self.label] 

        file_code = file_path[:-4]
        try:
            imgs = np.load(file_code + '.npy', allow_pickle=True)  # shape depends on how your .npy was saved
        except:
            imgs = np.zeros((50,25,3))
            print('Failed ',file_code)
        N, T, _ = imgs.shape
        imgs = imgs.reshape((N, T, 3)).transpose(2, 0, 1)


        # If you have a transform/pipeline, apply it
        if self.transform is not None:
            imgs = self.transform(imgs)

        # Return dict to match your requested "imgs" and "label"
        return {
            "imgs": imgs,
            "label": label
        }


def custom_collate_fn(batch):
    imgs = [item['imgs'] for item in batch]
    max_frames = max(item.shape[1] for item in imgs)

    padded_imgs = []
    lengths = []  # optional, to keep track of original lengths
    for img in imgs:
        dims, T, joints = img.shape
        lengths.append(T)
        if T < max_frames:
            # pad at the end along frame dimension
            pad = torch.zeros((dims, max_frames - T, joints))
            padded_tensor = torch.cat([torch.Tensor(img), pad], dim=1)
        else:
            padded_tensor = torch.Tensor(img)
        padded_imgs.append(padded_tensor)
        
        labels = torch.tensor([item["label"] for item in batch])
    # Stack to get shape (batch_size, max_frames, 11, 3)
    return {"imgs": torch.stack(padded_imgs), "label": labels}, lengths

def build_dataloader(df,
                     batch_size=8,
                     num_workers=1,
                     use_distributed=False,
                     custom_collate=True,
                     label='ACTION1_ID',
                     seed=42):
    """
    Args:
        csv_train (str): Path to the CSV for training (has [File, Function]).
        csv_val   (str): Path to the CSV for validation (has [File, Function]).
        batch_size (int): Batch size for both train/val
        num_workers (int): Number of worker processes
        use_distributed (bool): If True, use torch.distributed sampler
        custom_collate (bool): If True, use mmcv_collate. Otherwise, default_collate.

    Returns:
        train_data, val_data, train_loader, val_loader
    """
    df = df.dropna(subset=[label])

    df = df[~df[label].isin([6, 7])]

    train_df, val_df = train_test_split(
    df, 
    test_size=0.2,          
    random_state=seed,      
    stratify=df[label]    
    )

    train_data = CSVNpyDataset(train_df, label)
    val_data   = CSVNpyDataset(val_df, label)

    if use_distributed:
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        sampler_train = torch.utils.data.DistributedSampler(
            train_data, num_replicas=world_size, rank=rank, shuffle=True
        )

        val_indices = np.arange(rank, len(val_data), world_size)
        sampler_val = SubsetRandomSampler(val_indices)
    else:
        sampler_train = None
        sampler_val = None

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=(sampler_train is None),  # only shuffle if no distributed sampler
        sampler=sampler_train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=custom_collate_fn,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler_val,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=custom_collate_fn,
    )

    return train_loader, val_loader

