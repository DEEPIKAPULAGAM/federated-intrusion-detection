import sys

import flwr as fl
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset

from model import MLP


# ============================================================
# 1. SETTINGS
# ============================================================

client_number = sys.argv[1]

file_path = rf"C:\Users\DELL\Downloads\client_{client_number}.csv"

pc_columns = [f"PC{i}" for i in range(1, 150)]

chunk_size = 10000
batch_size = 256
local_epochs = 3
learning_rate = 0.001


# ============================================================
# 2. COUNT CLIENT RECORDS
# ============================================================

num_samples = sum(
    len(chunk)
    for chunk in pd.read_csv(
        file_path,
        usecols=["Label"],
        chunksize=100000
    )
)

print(f"Client {client_number} records: {num_samples}")


# ============================================================
# 3. LOAD CLIENT DATA IN CHUNKS
# ============================================================

class CSVDataset(IterableDataset):

    def __init__(self, file_path, chunk_size=10000):
        self.file_path = file_path
        self.chunk_size = chunk_size

    def __iter__(self):

        for chunk in pd.read_csv(
            self.file_path,
            usecols=pc_columns + ["Label"],
            chunksize=self.chunk_size
        ):

            X = torch.tensor(
                chunk[pc_columns].values,
                dtype=torch.float32
            )

            y = torch.tensor(
                chunk["Label"].values,
                dtype=torch.float32
            )

            for i in range(len(chunk)):
                yield X[i], y[i]


dataset = CSVDataset(
    file_path=file_path,
    chunk_size=chunk_size
)

loader = DataLoader(
    dataset,
    batch_size=batch_size
)


# ============================================================
# 4. CREATE MODEL AND TRAINING SETUP
# ============================================================

model = MLP()

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate
)

print("\nClient model:")
print(model)


# ============================================================
# 5. GET AND SET MODEL PARAMETERS
# ============================================================

def get_parameters(model):
    return [
        parameter.detach().cpu().numpy()
        for parameter in model.parameters()
    ]


def set_parameters(model, parameters):

    with torch.no_grad():

        for parameter, new_parameter in zip(
            model.parameters(),
            parameters
        ):

            parameter.copy_(
                torch.tensor(
                    new_parameter,
                    dtype=parameter.dtype,
                    device=parameter.device
                )
            )


# ============================================================
# 6. FLOWER CLIENT
# ============================================================

class FlowerClient(fl.client.NumPyClient):

    def get_parameters(self, config):
        return get_parameters(model)

    def fit(self, parameters, config):

        # Receive global model weights from the server
        set_parameters(model, parameters)

        model.train()

        # Train the model locally
        for epoch in range(local_epochs):

            total_loss = 0.0
            num_batches = 0

            for batch_X, batch_y in loader:

                outputs = model(batch_X).squeeze(1)

                loss = criterion(outputs, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            average_loss = (
                total_loss / num_batches
                if num_batches > 0
                else 0.0
            )

            print(
                f"Client {client_number} | "
                f"Epoch {epoch + 1}/{local_epochs} | "
                f"Loss: {average_loss:.6f}"
            )

        # Send locally trained weights and sample count to server
        return get_parameters(model), num_samples, {}

    def evaluate(self, parameters, config):

        # Client-side evaluation is not used.
        # The server evaluates the global model on validation.csv.
        return 0.0, num_samples, {}


# ============================================================
# 7. START FLOWER CLIENT
# ============================================================

fl.client.start_client(
    server_address="127.0.0.1:8080",
    client=FlowerClient().to_client(),
    insecure=True
)