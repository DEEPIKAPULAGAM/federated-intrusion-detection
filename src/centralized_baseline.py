import torch
import torch.nn as nn
import pandas as pd

from torch.utils.data import TensorDataset, DataLoader
from model import MLP


# -----------------------------
# Settings
# -----------------------------

BATCH_SIZE = 256
EPOCHS = 3
LEARNING_RATE = 0.001


# -----------------------------
# Client files
# -----------------------------

client_files = [
    r"C:\Users\DELL\Downloads\client_1.csv",
    r"C:\Users\DELL\Downloads\client_2.csv",
    r"C:\Users\DELL\Downloads\client_3.csv"
]


# -----------------------------
# Create model
# -----------------------------

model = MLP()

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# -----------------------------
# PCA columns
# -----------------------------

pc_columns = [f"PC{i}" for i in range(1, 150)]

# -----------------------------
# Centralized training
# -----------------------------

model.train()

for epoch in range(EPOCHS):

    total_loss = 0
    total_samples = 0

    for file in client_files:

        print(f"\nTraining on: {file}")

        df = pd.read_csv(file)

        X = torch.tensor(
            df[pc_columns].values,
            dtype=torch.float32
        )

        y = torch.tensor(
            df["Label"].values,
            dtype=torch.float32
        )

        dataset = TensorDataset(X, y)

        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True
        )

        for batch_X, batch_y in loader:

            outputs = model(batch_X).squeeze(1)

            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += loss.item() * len(batch_y)
            total_samples += len(batch_y)

        del df, X, y, dataset, loader

    average_loss = total_loss / total_samples

    print(
        f"\nEpoch {epoch + 1}/{EPOCHS}, "
        f"Average Loss: {average_loss:.6f}"
    )

# -----------------------------
# Validation
# -----------------------------

print("\nLoading validation data...")

validation_path = r"C:\Users\DELL\Downloads\validation.csv"

validation_df = pd.read_csv(validation_path)

X_val = torch.tensor(
    validation_df[pc_columns].values,
    dtype=torch.float32
)

y_val = torch.tensor(
    validation_df["Label"].values,
    dtype=torch.float32
)

validation_dataset = TensorDataset(X_val, y_val)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# -----------------------------
# Evaluate model
# -----------------------------

model.eval()

correct = 0
total = 0

all_predictions = []
all_labels = []

with torch.no_grad():

    for batch_X, batch_y in validation_loader:

        outputs = model(batch_X).squeeze(1)

        probabilities = torch.sigmoid(outputs)

        predictions = (probabilities >= 0.5).float()

        correct += (predictions == batch_y).sum().item()
        total += len(batch_y)

        all_predictions.extend(predictions.numpy())
        all_labels.extend(batch_y.numpy())


accuracy = correct / total

print("\nCentralized Validation Accuracy:", accuracy)

accuracy = correct / total

print("\nCentralized Validation Accuracy:", accuracy)

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

print("\nConfusion Matrix:")
print(confusion_matrix(all_labels, all_predictions))

print("\nClassification Report:")
print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=["Normal", "Attack"],
        digits=4
    )
)