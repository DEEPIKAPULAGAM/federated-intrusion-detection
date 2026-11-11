
import csv

import torch
import torch.nn as nn
import pandas as pd
import flwr as fl

from model import MLP


# ============================================================
# SETTINGS
# ============================================================

VALIDATION_FILE = r"C:\Users\Dell\Downloads\validation.csv"

BATCH_SIZE = 256
CHUNK_SIZE = 10000

METRICS_FILE = "noniid_fedprox_metrics_100.csv"

# MODEL CHECKPOINT FILES
LATEST_MODEL_FILE = "noniid_fedprox_latest.pth"
FINAL_MODEL_FILE = "noniid_fedprox_round100.pth"

PC_COLUMNS = [f"PC{i}" for i in range(1, 150)]


# ============================================================
# GLOBAL MODEL
# ============================================================

global_model = MLP()

print("\nFedProx Global Model:")
print(global_model)


# ============================================================
# CREATE METRICS CSV
# ============================================================

with open(METRICS_FILE, "w", newline="") as file:
    writer = csv.writer(file)

    writer.writerow([
        "Round",
        "Loss",
        "Accuracy",
        "Precision",
        "Recall",
        "F1"
    ])


# ============================================================
# GLOBAL MODEL VALIDATION
# ============================================================

def evaluate_global_model(server_round, parameters, config):

    print(
        f"\n========== FEDPROX VALIDATION - ROUND {server_round} =========="
    )

    # Load aggregated parameters into global model
    state_dict = global_model.state_dict()

    if len(parameters) != len(state_dict):
        raise ValueError("Model parameter count does not match.")

    for (key, old_value), array in zip(state_dict.items(), parameters):

        tensor = torch.as_tensor(
            array,
            dtype=old_value.dtype
        )

        if tensor.shape != old_value.shape:
            raise ValueError(f"Shape mismatch for {key}")

        state_dict[key] = tensor

    global_model.load_state_dict(state_dict)
    global_model.eval()

    criterion = nn.BCEWithLogitsLoss(reduction="sum")

    total_loss = 0.0
    total_samples = 0

    true_negative = 0
    false_positive = 0
    false_negative = 0
    true_positive = 0

    # --------------------------------------------------------
    # READ VALIDATION DATA IN CHUNKS
    # --------------------------------------------------------

    for chunk in pd.read_csv(
        VALIDATION_FILE,
        usecols=PC_COLUMNS + ["Label"],
        chunksize=CHUNK_SIZE,
        low_memory=False
    ):

        X = torch.tensor(
            chunk[PC_COLUMNS].values,
            dtype=torch.float32
        )

        y = torch.tensor(
            chunk["Label"].values,
            dtype=torch.float32
        )

        # ----------------------------------------------------
        # PROCESS EACH CHUNK IN BATCHES
        # ----------------------------------------------------

        for start in range(0, len(X), BATCH_SIZE):

            end = start + BATCH_SIZE

            batch_X = X[start:end]
            batch_y = y[start:end]

            with torch.no_grad():

                logits = global_model(batch_X).squeeze(1)

                loss = criterion(logits, batch_y)
                total_loss += loss.item()

                probabilities = torch.sigmoid(logits)

                predictions = (
                    probabilities >= 0.5
                ).float()

            # Update confusion matrix

            true_negative += (
                ((predictions == 0) & (batch_y == 0))
                .sum()
                .item()
            )

            false_positive += (
                ((predictions == 1) & (batch_y == 0))
                .sum()
                .item()
            )

            false_negative += (
                ((predictions == 0) & (batch_y == 1))
                .sum()
                .item()
            )

            true_positive += (
                ((predictions == 1) & (batch_y == 1))
                .sum()
                .item()
            )

            total_samples += len(batch_y)

    # ========================================================
    # CALCULATE METRICS
    # ========================================================

    if total_samples == 0:
        raise ValueError("Validation dataset is empty.")

    validation_loss = total_loss / total_samples

    accuracy = (
        (true_positive + true_negative)
        / total_samples
    )

    precision = (
        true_positive / (true_positive + false_positive)
        if (true_positive + false_positive) > 0
        else 0.0
    )

    recall = (
        true_positive / (true_positive + false_negative)
        if (true_positive + false_negative) > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    # ========================================================
    # PRINT METRICS
    # ========================================================

    print(f"Validation Loss : {validation_loss:.6f}")
    print(f"Accuracy        : {accuracy:.4f}")
    print(f"Precision       : {precision:.4f}")
    print(f"Recall          : {recall:.4f}")
    print(f"F1 Score        : {f1:.4f}")

    print("\nConfusion Matrix:")

    print(
        f"TN = {true_negative}, "
        f"FP = {false_positive}, "
        f"FN = {false_negative}, "
        f"TP = {true_positive}"
    )

    print("====================================================\n")

    # ========================================================
    # SAVE VALIDATION METRICS
    # ========================================================

    with open(METRICS_FILE, "a", newline="") as file:

        writer = csv.writer(file)

        writer.writerow([
            server_round,
            validation_loss,
            accuracy,
            precision,
            recall,
            f1
        ])

    # ========================================================
    # SAVE MODEL CHECKPOINTS
    # ========================================================

    # Save latest global model after every validation round

    torch.save(
        global_model.state_dict(),
        LATEST_MODEL_FILE
    )

    print(
        f"Latest model checkpoint saved: {LATEST_MODEL_FILE}"
    )

    # Save final global model at round 100

    if server_round == 100:

        torch.save(
            global_model.state_dict(),
            FINAL_MODEL_FILE
        )

        print(
            f"Final model checkpoint saved: {FINAL_MODEL_FILE}"
        )

    return validation_loss, {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# ============================================================
# FEDAVG AGGREGATION
# ============================================================

strategy = fl.server.strategy.FedAvg(

    fraction_fit=1.0,

    min_fit_clients=3,

    min_available_clients=3,

    evaluate_fn=evaluate_global_model,

    fraction_evaluate=0.0
)


# ============================================================
# START FEDPROX SERVER
# ============================================================

print("\nStarting FedProx Server...")
print("Aggregation: FedAvg")
print("FedProx MU: 0.01")
print("Number of clients: 3")
print("Number of rounds: 100")

fl.server.start_server(

    server_address="127.0.0.1:8080",

    config=fl.server.ServerConfig(
        num_rounds=100
    ),

    strategy=strategy,

    grpc_max_message_length=2147483647
)
