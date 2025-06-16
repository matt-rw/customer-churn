#!/usr/bin/env python

from sklearn.metrics import confusion_matrix, classification_report
import torch
from torch.utils.data import DataLoader

from churn.model import ChurnModel
from churn.dataset import load_datasets


# Load data
_, val_ds = load_datasets()
val_loader = DataLoader(val_ds, batch_size=32)

# Load model
input_dim = val_ds[0][0].shape[0]
model = ChurnModel(input_dim)
model.load_state_dict(torch.load('churn_model.pt'))
model.eval()


correct = 0
total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for X_batch, y_batch in val_loader:
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        
        # For accuracy
        correct += (predicted == y_batch).sum().item()
        total += y_batch.size(0)

        # For Confusion Matrix and Classification Report
        all_preds.extend(predicted.tolist())
        all_labels.extend(y_batch.tolist())

accuracy = correct / total
print(f'Validation Accuracy: {accuracy:.2%}')

print('\nConfusion Matrix:')
print(confusion_matrix(all_labels, all_preds))

print('\nClassification Report:')
print(classification_report(all_labels, all_preds, digits=3))

## For binary classification:
## 
##           Predicted
##           | 0  | 1  |
##          -----------
## Actual  0 | TN | FP |
##         1 | FN | TP |


