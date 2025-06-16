#!/usr/bin/env python

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from churn.dataset import load_datasets
from churn.model import ChurnModel


# Load data
train_ds, val_ds = load_datasets()
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

# Initialize model
x_sample, _ = train_ds[0]
input_dim = x_sample.shape[0]

model = ChurnModel(input_dim)

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 10

# Train loop
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f'Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}')

# Save model
torch.save(model.state_dict(), 'churn_model.pt')


