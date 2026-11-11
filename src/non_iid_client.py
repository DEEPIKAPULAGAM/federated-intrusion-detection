import csv
import os
import sys

import torch
import torch.nn as nn
import pandas as pd
import flwr as fl

from model import MLP
from torch.utils.data import IterableDataset, DataLoader


# ============================================================
# 1. Identify the client
# ============================================================

client_number = sys.argv[1]


# ============================================================
# 2. Non-IID client file
# ============================================================

file_path = rf"C:\federated-ids\non_iid\noniidclient_{client_number}.csv"

# Calculate actual number of records
num_samples = sum(
    len(chunk)
    for chunk in pd.read_csv(
        file_path,
        usecols=["Label"],
        chunksize=100000
    )
)

print(f"\nStarting Non-IID Client {client_number}")
print(f"Data file: {file_path}")
print(f"Number of samples: {num_samples:,}")


# ============================================================
# 3. Select PCA input features
# ============================================================

pc_columns = [f"PC{i}" for i in range(1, 150)]


# ============================================================
# 4. Chunked CSV Dataset
# ============================================================

class CSVDataset(IterableDataset):

    def __init__(self, file_path, chunk_size=10000):
        self.file_path = file_path
        self.chunk_size = chunk_size

    def __iter__(self):

        for chunk in pd.read_csv(
            self.file_path,
            usecols=pc_columns + ["Label"],
            chunksize=self.chunk_size,
            low_memory=False
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


dataset = CSVDataset(file_path)

loader = DataLoader(
    dataset,
    batch_size=256
)


# ============================================================
# 5. Create the model
# ============================================================

model = MLP()

print("\nModel:")
print(model)


# ============================================================
# 6. Training setup
# ============================================================

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# ============================================================
# 7. Get model parameters
# ============================================================

def get_parameters(model):

    return [
        val.detach().cpu().numpy()
        for val in model.parameters()
    ]


# ============================================================
# 8. Set model parameters
# ============================================================

def set_parameters(model, parameters):

    with torch.no_grad():

        for param, new_param in zip(
            model.parameters(),
            parameters
        ):

            param.copy_(
                torch.tensor(
                    new_param,
                    dtype=param.dtype
                )
            )


# ============================================================
# 9. Model Drift CSV setup
# ============================================================

round_number = 0

drift_file = (
    f"drift_noniid_client_{client_number}.csv"
)


# ============================================================
# 10. Flower Client
# ============================================================

class FlowerClient(fl.client.NumPyClient):

    def get_parameters(self, config):

        return get_parameters(model)


    def set_parameters(self, parameters):

        set_parameters(model, parameters)


    def fit(self, parameters, config):

        global round_number

        round_number += 1

        # -----------------------------------------
        # Receive the current global model
        # -----------------------------------------

        self.set_parameters(parameters)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.001
        )

        # Save a copy of global model parameters
        global_parameters = [
            param.detach().clone()
            for param in model.parameters()
        ]

        model.train()

        # -----------------------------------------
        # Local training: 3 epochs
        # -----------------------------------------

        for epoch in range(3):

            total_loss = 0
            num_batches = 0

            for batch_X, batch_y in loader:

                # Forward pass
                outputs = model(batch_X).squeeze(1)

                # Calculate classification loss
                loss = criterion(
                    outputs,
                    batch_y
                )

                # Clear old gradients
                optimizer.zero_grad()

                # Backpropagation
                loss.backward()

                # Update weights
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            average_loss = total_loss / num_batches

            print(
                f"Client {client_number} | "
                f"Epoch {epoch + 1}/3 | "
                f"Loss: {average_loss:.6f}"
            )

        # ==================================================
        # 11. Calculate Model Drift
        # ==================================================

        local_parameters = [
            param.detach().clone()
            for param in model.parameters()
        ]

        # Calculate L2 distance
        drift_numerator = torch.sqrt(
            sum(
                torch.sum(
                    (local_param - global_param) ** 2
                )
                for local_param, global_param in zip(
                    local_parameters,
                    global_parameters
                )
            )
        )

        drift_denominator = torch.sqrt(
            sum(
                torch.sum(global_param ** 2)
                for global_param in global_parameters
            )
        )

        # Relative model drift
        drift = (
            drift_numerator /
            (drift_denominator + 1e-12)
        ).item()

        # ==================================================
        # 12. Save Model Drift to CSV
        # ==================================================

        file_exists = os.path.isfile(drift_file)

        with open(
            drift_file,
            "a",
            newline=""
        ) as file:

            writer = csv.writer(file)

            if not file_exists:

                writer.writerow([
                    "Round",
                    "Client",
                    "Model_Drift"
                ])

            writer.writerow([
                round_number,
                client_number,
                drift
            ])

        print(
            f"Client {client_number} | "
            f"Round {round_number} | "
            f"Model Drift: {drift:.6f}"
        )

        # -----------------------------------------
        # Send updated model parameters to server
        # -----------------------------------------

        return (
            get_parameters(model),
            num_samples,
            {"model_drift": drift}
        )


    def evaluate(self, parameters, config):

        return 0.0, num_samples, {}


# ============================================================
# 13. Start Flower Client
# ============================================================

print("\nConnecting to Non-IID Flower server...")

fl.client.start_client(
    server_address="127.0.0.1:8080",
    client=FlowerClient().to_client(),
    insecure=True,
)
