import flwr as fl
import pandas as pd
import torch
import torch.nn as nn
import csv
import os

from model import MLP


# ============================================================
# 1. SETTINGS
# ============================================================

VALIDATION_FILE = r"C:\Users\Dell\Downloads\validation.csv"

BATCH_SIZE = 256
CHUNK_SIZE = 10000
NUM_ROUNDS = 100

PC_COLUMNS = [f"PC{i}" for i in range(1, 150)]

MODEL_OUTPUT_FILE = f"iid_fedavg_round_{NUM_ROUNDS}.pth"
METRICS_OUTPUT_FILE = "iid_fedavg_metrics_100.csv"


# ============================================================
# 2. GLOBAL MODEL
# ============================================================

model = MLP()

print("Global model:")
print(model)


# ============================================================
# 3. VALIDATION FUNCTION
# ============================================================

def evaluate_global_model(server_round, parameters, config):

    print(f"\n========== VALIDATION - ROUND {server_round} ==========")

    ndarrays = parameters
    state_dict = model.state_dict()

    for key, array in zip(state_dict.keys(), ndarrays):
        state_dict[key] = torch.tensor(
            array,
            dtype=state_dict[key].dtype
        )

    model.load_state_dict(state_dict)
    model.eval()

    criterion = nn.BCEWithLogitsLoss(reduction="sum")

    total_loss = 0.0
    total_samples = 0

    true_negative = 0
    false_positive = 0
    false_negative = 0
    true_positive = 0

    for chunk in pd.read_csv(
        VALIDATION_FILE,
        usecols=PC_COLUMNS + ["Label"],
        chunksize=CHUNK_SIZE
    ):

        X = torch.tensor(
            chunk[PC_COLUMNS].values,
            dtype=torch.float32
        )

        y = torch.tensor(
            chunk["Label"].values,
            dtype=torch.float32
        )

        for start in range(0, len(X), BATCH_SIZE):

            end = start + BATCH_SIZE

            batch_X = X[start:end]
            batch_y = y[start:end]

            with torch.no_grad():
                logits = model(batch_X).squeeze(1)

                loss = criterion(logits, batch_y)
                total_loss += loss.item()

                probabilities = torch.sigmoid(logits)
                predictions = (probabilities >= 0.5).float()

            true_negative += (
                ((predictions == 0) & (batch_y == 0)).sum().item()
            )

            false_positive += (
                ((predictions == 1) & (batch_y == 0)).sum().item()
            )

            false_negative += (
                ((predictions == 0) & (batch_y == 1)).sum().item()
            )

            true_positive += (
                ((predictions == 1) & (batch_y == 1)).sum().item()
            )

            total_samples += len(batch_y)

    validation_loss = total_loss / total_samples if total_samples else 0.0

    accuracy = (
        (true_positive + true_negative) / total_samples
        if total_samples else 0.0
    )

    precision = (
        true_positive / (true_positive + false_positive)
        if (true_positive + false_positive) else 0.0
    )

    recall = (
        true_positive / (true_positive + false_negative)
        if (true_positive + false_negative) else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) else 0.0
    )

    print(f"Validation Loss : {validation_loss:.6f}")
    print(f"Accuracy        : {accuracy:.4f}")
    print(f"Precision       : {precision:.4f}")
    print(f"Recall          : {recall:.4f}")
    print(f"F1 Score        : {f1:.4f}")
    print(
        f"TN={true_negative}, FP={false_positive}, "
        f"FN={false_negative}, TP={true_positive}"
    )

    # Append this round's validation metrics to the CSV
    file_exists = os.path.isfile(METRICS_OUTPUT_FILE)

    with open(METRICS_OUTPUT_FILE, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)

        if not file_exists:
            writer.writerow([
                "round", "validation_loss", "accuracy",
                "precision", "recall", "f1"
            ])

        writer.writerow([
            server_round,
            validation_loss,
            accuracy,
            precision,
            recall,
            f1
        ])

    print(f"Metrics saved to {METRICS_OUTPUT_FILE}")
    print("====================================================\n")

    return validation_loss, {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# ============================================================
# 4. FEDAVG STRATEGY AND MODEL SAVING
# ============================================================

class SaveFinalFedAvg(fl.server.strategy.FedAvg):

    def aggregate_fit(self, server_round, results, failures):

        aggregated_parameters, metrics = super().aggregate_fit(
            server_round,
            results,
            failures
        )

        # Save the global model after the final round
        if (
            server_round == NUM_ROUNDS
            and aggregated_parameters is not None
        ):

            arrays = fl.common.parameters_to_ndarrays(
                aggregated_parameters
            )

            state_dict = model.state_dict()

            for key, array in zip(state_dict.keys(), arrays):
                state_dict[key] = torch.tensor(
                    array,
                    dtype=state_dict[key].dtype
                )

            model.load_state_dict(state_dict)

            torch.save(model.state_dict(), MODEL_OUTPUT_FILE)

            print(f"\nSaved final global model: {MODEL_OUTPUT_FILE}")

        return aggregated_parameters, metrics


# ============================================================
# 5. CREATE FEDAVG STRATEGY
# ============================================================

strategy = SaveFinalFedAvg(
    fraction_fit=1.0,
    min_fit_clients=3,
    min_available_clients=3,
    evaluate_fn=evaluate_global_model,
    fraction_evaluate=0.0
)


# ============================================================
# 6. START FLOWER SERVER
# ============================================================

fl.server.start_server(
    server_address="127.0.0.1:8080",
    grpc_max_message_length=2_147_483_647,
    config=fl.server.ServerConfig(
        num_rounds=NUM_ROUNDS
    ),
    strategy=strategy
)