
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np

from model import MLP

# Device
device = torch.device("cpu")

# Checkpoints
MODELS = {
    "Non-IID FedAvg": "noniid_fedavg_round100.pth",
    "Non-IID FedProx": "noniid_fedprox_latest.pth"
}

# Client datasets
CLIENT_FILES = {
    "Client 1": "non_iid/noniidclient_1.csv",
    "Client 2": "non_iid/noniidclient_2.csv",
    "Client 3": "non_iid/noniidclient_3.csv"
}

FEATURES = [f"PC{i}" for i in range(1, 150)]
BATCH_SIZE = 256


def calculate_loss(model, file_path):

    total_loss = 0.0
    total_samples = 0

    for chunk in pd.read_csv(file_path, chunksize=10000):

        X = chunk[FEATURES].values.astype(np.float32)
        y = chunk["Label"].values.astype(np.float32)

        for start in range(0, len(X), BATCH_SIZE):

            end = start + BATCH_SIZE

            batch_X = torch.tensor(X[start:end])
            batch_y = torch.tensor(y[start:end])

            with torch.no_grad():
                logits = model(batch_X).squeeze(1)

                loss = F.binary_cross_entropy_with_logits(
                    logits,
                    batch_y,
                    reduction="sum"
                )

            total_loss += loss.item()
            total_samples += len(batch_y)

    return total_loss / total_samples, total_samples


results = []

for model_name, checkpoint in MODELS.items():

    print("\n" + "=" * 55)
    print(model_name)
    print("=" * 55)

    model = MLP().to(device)

    state_dict = torch.load(
        checkpoint,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(state_dict)
    model.eval()

    for client_name, file_path in CLIENT_FILES.items():

        loss, count = calculate_loss(model, file_path)

        print(
            f"{client_name}: Loss = {loss:.6f}, "
            f"Samples = {count}"
        )

        results.append({
            "Model": model_name,
            "Client": client_name,
            "Test_Loss": loss,
            "Samples": count
        })


df = pd.DataFrame(results)

print("\nFINAL CLIENT-WISE LOSS COMPARISON")
print(df.pivot(
    index="Client",
    columns="Model",
    values="Test_Loss"
).to_string())

df.to_csv("client_wise_loss_comparison.csv", index=False)

print("\nSaved: client_wise_loss_comparison.csv")
