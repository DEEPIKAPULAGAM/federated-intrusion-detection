
import csv
import flwr as fl
import torch
import torch.nn as nn
import pandas as pd

from model import MLP


# ============================================================
# SETTINGS
# ============================================================

VALIDATION_FILE = r"C:\Users\Dell\Downloads\validation.csv"

BATCH_SIZE = 256
CHUNK_SIZE = 10000

METRICS_FILE = "noniid_fedavg_metrics_100.csv"

# MODEL CHECKPOINT FILES
LATEST_MODEL_FILE = "noniid_fedavg_latest.pth"
FINAL_MODEL_FILE = "noniid_fedavg_round100.pth"


# ============================================================
# GLOBAL MODEL
# ============================================================

model = MLP()

print("Non-IID Global Model:")
print(model)


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
# VALIDATION FUNCTION
# ============================================================

def evaluate_global_model(server_round, parameters, config):

    print(f"\n========== NON-IID VALIDATION - ROUND {server_round} ==========")

    # Flower provides the parameters as NumPy arrays
    ndarrays = parameters

    # Put the global parameters into the model
    state_dict = model.state_dict()

    for key, array in zip(state_dict.keys(), ndarrays):
        state_dict[key] = torch.tensor(
            array,
            dtype=state_dict[key].dtype
        )

    model.load_state_dict(state_dict)

    # Evaluation mode
    model.eval()

    # Loss function
    criterion = nn.BCEWithLogitsLoss(reduction="sum")

    # PCA input columns
    pc_columns = [f"PC{i}" for i in range(1, 150)]

    total_loss = 0.0
    total_samples = 0

    # Confusion matrix
    true_negative = 0
    false_positive = 0
    false_negative = 0
    true_positive = 0

    # --------------------------------------------------------
    # Read validation data in chunks
    # --------------------------------------------------------

    for chunk in pd.read_csv(
        VALIDATION_FILE,
        usecols=pc_columns + ["Label"],
        chunksize=CHUNK_SIZE
    ):

        X = torch.tensor(
            chunk[pc_columns].values,
            dtype=torch.float32
        )

        y = torch.tensor(
            chunk["Label"].values,
            dtype=torch.float32
        )

        # ----------------------------------------------------
        # Process the chunk in batches
        # ----------------------------------------------------

        for start in range(0, len(X), BATCH_SIZE):

            end = start + BATCH_SIZE

            batch_X = X[start:end]
            batch_y = y[start:end]

            with torch.no_grad():

                # Model output = logits
                logits = model(batch_X).squeeze(1)

                # Calculate validation loss
                loss = criterion(logits, batch_y)

                total_loss += loss.item()

                # Convert logits to probabilities
                probabilities = torch.sigmoid(logits)

                # Probability >= 0.5 -> Attack
                predictions = (
                    probabilities >= 0.5
                ).float()

            # ------------------------------------------------
            # Update confusion matrix
            # ------------------------------------------------

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
        true_positive
        / (true_positive + false_positive)
        if (true_positive + false_positive) > 0
        else 0
    )

    recall = (
        true_positive
        / (true_positive + false_negative)
        if (true_positive + false_negative) > 0
        else 0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    # ========================================================
    # PRINT RESULTS
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
    # SAVE METRICS TO CSV
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
    # SAVE MODEL WEIGHTS
    # ========================================================

    # Save latest global model after every validation round
    torch.save(
        model.state_dict(),
        LATEST_MODEL_FILE
    )

    print(f"Latest model checkpoint saved: {LATEST_MODEL_FILE}")

    # Save final model explicitly at round 100
    if server_round == 100:

        torch.save(
            model.state_dict(),
            FINAL_MODEL_FILE
        )

        print(
            f"Final model checkpoint saved: {FINAL_MODEL_FILE}"
        )

    return validation_loss, {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ============================================================
# FEDAVG STRATEGY
# ============================================================

strategy = fl.server.strategy.FedAvg(

    # All 3 clients participate
    fraction_fit=1.0,

    min_fit_clients=3,

    min_available_clients=3,

    # Server-side validation
    evaluate_fn=evaluate_global_model,

    # No client-side validation
    fraction_evaluate=0.0,
)


# ============================================================
# START FLOWER SERVER
# ============================================================

print("\nStarting Non-IID FedAvg Server...")
print("Number of rounds: 100")

fl.server.start_server(

    server_address="127.0.0.1:8080",

    grpc_max_message_length=2_147_483_647,

    config=fl.server.ServerConfig(
        num_rounds=100
    ),

    strategy=strategy,
)
