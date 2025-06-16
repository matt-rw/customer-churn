import torch
from torch.utils.data import Dataset, DataLoader

from churn.preprocess import load_and_preprocess


class ChurnDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def load_datasets():
    X_train, X_val, y_train, y_val = load_and_preprocess()

    train_ds = ChurnDataset(X_train, y_train)
    val_ds = ChurnDataset(X_val, y_val)

    return train_ds, val_ds

# train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
