
import csv
import os

import torch
import torch.nn as nn
import pandas as pd

from model import MLP


# ============================================================
# SETTINGS
# ============================================================

TEST_FILE = r"C:\federated-ids\test.csv"

BATCH_SIZE = 256
CHUNK_SIZE = 10000

PC_COLUMNS = [f"PC{i}" for i in range(1, 150)]

# All three trained models
MODELS = {
    "IID FedAvg": "iid_fedavg_round_100.pth",

    "Non-IID FedAvg": "noniid_fedavg_round100.pth",

    "Non-IID FedProx": "noniid_fedprox_latest.pth"
}

# Output files
METRICS_FILE = "all_models_test_metrics.csv"

CONFUSION_FILE = "all_models_confusion_matrix.csv"

PREDICTIONS_FOLDER = "test_predictions"


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

print("\nChecking required files...")

if not os.path.exists(TEST_FILE):
    raise FileNotFoundError(
        f"Test dataset not found: {TEST_FILE}"
    )

for model_name, model_file in MODELS.items():

    if not os.path.exists(model_file):
        raise FileNotFoundError(
            f"Checkpoint missing for {model_name}: {model_file}"
        )

print("All required files found!")


# ============================================================
# CREATE OUTPUT FOLDER
# ============================================================

os.makedirs(PREDICTIONS_FOLDER, exist_ok=True)


# ============================================================
# LOSS FUNCTION
# ============================================================

criterion = nn.BCEWithLogitsLoss(reduction="sum")


# ============================================================
# MODEL TESTING FUNCTION
# ============================================================

def test_model(model_name, model_file):

    print("\n" + "=" * 55)
    print(f"TESTING: {model_name}")
    print("=" * 55)

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    model = MLP()

    state_dict = torch.load(
        model_file,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(state_dict)
    model.eval()

    print("Model loaded:", model_file)

    # --------------------------------------------------------
    # INITIALIZE METRICS
    # --------------------------------------------------------

    total_loss = 0.0
    total_samples = 0

    true_negative = 0
    false_positive = 0
    false_negative = 0
    true_positive = 0

    prediction_file = os.path.join(
        PREDICTIONS_FOLDER,
        model_name.lower().replace(" ", "_") + "_predictions.csv"
    )

    # --------------------------------------------------------
    # READ TEST DATA IN CHUNKS
    # --------------------------------------------------------

    with open(prediction_file, "w", newline="") as file:

        writer = csv.writer(file)

        writer.writerow([
            "Actual_Label",
            "Predicted_Label",
            "Probability"
        ])

        for chunk in pd.read_csv(
            TEST_FILE,
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

            # ------------------------------------------------
            # PROCESS IN BATCHES
            # ------------------------------------------------

            for start in range(0, len(X), BATCH_SIZE):

                end = start + BATCH_SIZE

                batch_X = X[start:end]
                batch_y = y[start:end]

                with torch.no_grad():

                    logits = model(batch_X).squeeze(1)

                    loss = criterion(logits, batch_y)

                    total_loss += loss.item()

                    probabilities = torch.sigmoid(logits)

                    predictions = (
                        probabilities >= 0.5
                    ).float()

                # --------------------------------------------
                # UPDATE CONFUSION MATRIX
                # --------------------------------------------

                true_negative += (
                    ((predictions == 0) & (batch_y == 0))
                    .sum().item()
                )

                false_positive += (
                    ((predictions == 1) & (batch_y == 0))
                    .sum().item()
                )

                false_negative += (
                    ((predictions == 0) & (batch_y == 1))
                    .sum().item()
                )

                true_positive += (
                    ((predictions == 1) & (batch_y == 1))
                    .sum().item()
                )

                total_samples += len(batch_y)

                # --------------------------------------------
                # SAVE PREDICTIONS
                # --------------------------------------------

                writer.writerows(
                    zip(
                        batch_y.tolist(),
                        predictions.tolist(),
                        probabilities.tolist()
                    )
                )

    # ========================================================
    # CALCULATE FINAL METRICS
    # ========================================================

    if total_samples == 0:
        raise ValueError("Test dataset is empty.")

    test_loss = total_loss / total_samples

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
    # PRINT RESULTS
    # ========================================================

    print(f"\nModel: {model_name}")

    print(f"Test Loss : {test_loss:.6f}")
    print(f"Accuracy  : {accuracy:.6f}")
    print(f"Precision : {precision:.6f}")
    print(f"Recall    : {recall:.6f}")
    print(f"F1 Score  : {f1:.6f}")

    print("\nConfusion Matrix:")

    print(f"TN = {true_negative}")
    print(f"FP = {false_positive}")
    print(f"FN = {false_negative}")
    print(f"TP = {true_positive}")

    print("\nPredictions saved:", prediction_file)

    # ========================================================
    # RETURN RESULTS
    # ========================================================

    return {
        "Model": model_name,
        "Test_Loss": test_loss,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Total_Samples": total_samples,
        "TN": true_negative,
        "FP": false_positive,
        "FN": false_negative,
        "TP": true_positive
    }


# ============================================================
# TEST ALL THREE MODELS
# ============================================================

all_results = []

for model_name, model_file in MODELS.items():

    results = test_model(
        model_name,
        model_file
    )

    all_results.append(results)


# ============================================================
# SAVE COMPARISON METRICS
# ============================================================

results_df = pd.DataFrame(all_results)

results_df.to_csv(
    METRICS_FILE,
    index=False
)


# ============================================================
# SAVE COMPARISON CONFUSION MATRICES
# ============================================================

confusion_df = results_df[
    ["Model", "TN", "FP", "FN", "TP"]
]

confusion_df.to_csv(
    CONFUSION_FILE,
    index=False
)


# ============================================================
# FINAL COMPARISON
# ============================================================

print("\n\n")
print("=" * 75)
print("FINAL TEST PERFORMANCE COMPARISON")
print("=" * 75)

print(
    results_df[
        [
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1"
        ]
    ].to_string(index=False)
)

print("\nGenerated files:")

print("1.", METRICS_FILE)
print("2.", CONFUSION_FILE)
print("3.", PREDICTIONS_FOLDER)

print("\nAll model testing completed successfully!")
