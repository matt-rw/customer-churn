import torch.nn as nn


class ChurnModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # For binary classification
        )

    def forward(self, x):
        return self.network(x)
