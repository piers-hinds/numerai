import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def get_dataloaders(df, train_index, test_index, features, targets):
    train_dataset, val_dataset = get_datasets(df, train_index, test_index, features, targets)
    train_dataloader = DataLoader(train_dataset, batch_size=1)
    val_dataloader = DataLoader(val_dataset, batch_size=1)
    return train_dataloader, val_dataloader


def get_datasets(df, train_index, test_index, features, targets):
    train_dataset = EraDataset(df.iloc[train_index], features, targets)
    val_dataset = EraDataset(df.iloc[test_index], features, targets)
    return train_dataset, val_dataset


class EraDataset(Dataset):
    """
    Dataset for era batches
    """
    def __init__(self, df, features, targets):
        self.df = df
        self.features = features
        self.targets = targets
        self.eras = df.era.unique()

    def __len__(self):
        return len(self.eras)

    def __getitem__(self, idx):
        era = [self.eras[idx]]
        x = self.df.loc[self.df.era.isin(era), self.features].values.astype(np.float32) - 0.5
        y = self.df.loc[self.df.era.isin(era), self.targets].values.astype(np.float32)

        return torch.tensor(x), torch.tensor(y)
