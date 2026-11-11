import csv
import os
import sys

import torch
import pandas as pd
import flwr as fl

from model import MLP
from torch.utils.data import IterableDataset, DataLoader


# Get client number from command-line argument
client_number = int(sys.argv[1])

# FedProx configuration
MU = 0.01
round_number = 0

# CSV file for recording model drift
drift_file = f"drift_fedprox_client_{client_number}.csv"


# Dataset class
class UNSWDataset(IterableDataset):
    def __init__(self, file_path, chunksize=10000):
        self.file_path = file_path
        self.chunksize = chunksize

    def __iter__(self):
        for chunk in pd.read_csv(self.file_path, chunksize=self.chunksize):
            X = torch.tensor(
                chunk.iloc[:, :149].values,
                dtype=torch.float32
            )

            y = torch.tensor(
                chunk["Label"].values,
                dtype=torch.float32
            ).unsqueeze(1)

            for i in range(len(X)):
                yield X[i], y[i]


# Client class
class FlowerClient(fl.client.NumPyClient):

    def __init__(self):
        global client_number

        self.model = MLP()
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=0.001
        )

        self.file_path = (
            f"C:\\federated-ids\\non_iid\\"
            f"noniidclient_{client_number}.csv"
        )

        # Dynamically count client samples
        self.num_samples = sum(
            len(chunk)
            for chunk in pd.read_csv(
                self.file_path,
                chunksize=100000
            )
        )

        self.dataset = UNSWDataset(self.file_path)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=256
        )

    # Get model parameters
    def get_parameters(self, config):
        return [
            val.cpu().numpy()
            for val in self.model.state_dict().values()
        ]

    # Set model parameters
    def set_parameters(self, parameters):
        state_dict = self.model.state_dict()

        for key, val in zip(state_dict.keys(), parameters):
            state_dict[key] = torch.tensor(val)

        self.model.load_state_dict(state_dict)

    # Local training
    def fit(self, parameters, config):
        global round_number

        round_number += 1

        # Receive the current global model
        self.set_parameters(parameters)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=0.001
        )

        # Save a copy of global parameters before local training
        global_parameters = [
            param.detach().clone()
            for param in self.model.parameters()
        ]

        self.model.train()

        for epoch in range(3):
            for X, y in self.dataloader:
                self.optimizer.zero_grad()

                outputs = self.model(X)

                # Original BCE loss
                loss = self.criterion(outputs, y)

                # FedProx proximal term
                proximal_term = 0.0

                for local_param, global_param in zip(
                    self.model.parameters(),
                    global_parameters
                ):
                    proximal_term += torch.sum(
                        (local_param - global_param) ** 2
                    )

                loss = loss + (MU / 2) * proximal_term

                loss.backward()
                self.optimizer.step()

        # Calculate relative model drift after local training
        local_parameters = [
            param.detach().clone()
            for param in self.model.parameters()
        ]

        drift_numerator = torch.sqrt(
            sum(
                torch.sum((local_param - global_param) ** 2)
                for local_param, global_param
                in zip(local_parameters, global_parameters)
            )
        )

        drift_denominator = torch.sqrt(
            sum(
                torch.sum(global_param ** 2)
                for global_param in global_parameters
            )
        )

        drift = (
            drift_numerator / (drift_denominator + 1e-12)
        ).item()

        # Save drift value to CSV
        file_exists = os.path.isfile(drift_file)

        with open(drift_file, "a", newline="") as f:
            writer = csv.writer(f)

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
            f"FedProx | Client {client_number} | "
            f"Round {round_number} | Drift: {drift:.6f}"
        )

        return (
            self.get_parameters(config),
            self.num_samples,
            {"model_drift": drift}
        )

    # Evaluation
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        return 0.0, self.num_samples, {}


# Start Flower client
client = FlowerClient()

fl.client.start_client(
    server_address="127.0.0.1:8080",
    client=client.to_client()
)
