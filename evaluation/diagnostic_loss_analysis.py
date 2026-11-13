
import os
import sys
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

SRC_DIR = os.path.join(PROJECT_ROOT, "src")

sys.path.insert(0, SRC_DIR)

from model import MLP

TEST_FILE = os.path.join(PROJECT_ROOT, "test.csv")

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT, "results", "metrics"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# SETTINGS
# ============================================================

CHUNK_SIZE = 10000

FEATURES = [f"PC{i}" for i in range(1, 150)]

# Same two zero-based test-row indices for all models
BAD_ROWS = [17945, 161268]

MODELS = {
    "IID FedAvg": "iid_fedavg_round_100.pth",
    "Non-IID FedAvg": "noniid_fedavg_round100.pth",
    "Non-IID FedProx": "noniid_fedprox_latest.pth"
}

# ============================================================
# OUTPUT FILES
# ============================================================

COMPARISON_FILE = os.path.join(
    OUTPUT_DIR,
    "diagnostic_loss_comparison.csv"
)

EXCLUDED_DETAILS_FILE = os.path.join(
    OUTPUT_DIR,
    "excluded_samples_loss_details.csv"
)

# ============================================================
# CHECK FILES
# ============================================================

if not os.path.exists(TEST_FILE):
    raise FileNotFoundError(
        f"Test dataset not found: {TEST_FILE}"
    )

for model_name, checkpoint in MODELS.items():

    checkpoint_path = os.path.join(
        PROJECT_ROOT, checkpoint
    )

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint missing for {model_name}: "
            f"{checkpoint_path}"
        )

print("All required files found.")

# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\nUsing device:", device)

# ============================================================
# MODEL EVALUATION FUNCTION
# ============================================================

def evaluate_model(model_name, checkpoint):

    print("\n" + "=" * 60)
    print("EVALUATING:", model_name)
    print("=" * 60)

    model = MLP().to(device)

    checkpoint_path = os.path.join(
        PROJECT_ROOT, checkpoint
    )

    state = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(state)
    model.eval()

    total_loss = 0.0
    adjusted_loss = 0.0

    total_samples = 0
    excluded_loss = 0.0

    excluded_details = []

    with torch.no_grad():

        for chunk in pd.read_csv(
            TEST_FILE,
            usecols=FEATURES + ["Label"],
            chunksize=CHUNK_SIZE,
            low_memory=False
        ):

            X = torch.tensor(
                chunk[FEATURES].to_numpy(dtype=np.float32),
                dtype=torch.float32,
                device=device
            )

            y = torch.tensor(
                chunk["Label"].to_numpy(dtype=np.float32),
                dtype=torch.float32,
                device=device
            )

            logits = model(X).reshape(-1)

            losses = F.binary_cross_entropy_with_logits(
                logits,
                y,
                reduction="none"
            )

            losses_np = losses.cpu().numpy()
            logits_np = logits.cpu().numpy()
            labels_np = y.cpu().numpy()

            for i in range(len(losses_np)):

                global_index = total_samples + i
                sample_loss = float(losses_np[i])

                total_loss += sample_loss

                if global_index in BAD_ROWS:

                    excluded_loss += sample_loss

                    excluded_details.append({
                        "Model": model_name,
                        "Test_Row_Index": global_index,
                        "Actual_Label": int(labels_np[i]),
                        "Logit": float(logits_np[i]),
                        "Individual_Loss": sample_loss
                    })

                else:

                    adjusted_loss += sample_loss

            total_samples += len(losses_np)

    # --------------------------------------------------------
    # VALIDATE EXCLUDED ROWS
    # --------------------------------------------------------

    found_indices = {
        item["Test_Row_Index"]
        for item in excluded_details
    }

    if found_indices != set(BAD_ROWS):
        raise ValueError(
            f"Not all excluded rows were found for {model_name}. "
            f"Found: {sorted(found_indices)}"
        )

    # --------------------------------------------------------
    # CALCULATE AVERAGES
    # --------------------------------------------------------

    original_average = total_loss / total_samples

    adjusted_samples = total_samples - len(BAD_ROWS)

    adjusted_average = adjusted_loss / adjusted_samples

    excluded_percentage = (
        excluded_loss / total_loss * 100
        if total_loss > 0 else 0
    )

    print("Total samples:", total_samples)
    print("Original average loss:", original_average)
    print("Adjusted average loss:", adjusted_average)
    print("Excluded sample loss:", excluded_loss)
    print("Percentage of total loss from excluded samples:",
          excluded_percentage, "%")

    # --------------------------------------------------------
    # RETURN RESULTS
    # --------------------------------------------------------

    comparison_result = {
        "Model": model_name,
        "Original_Test_Loss": original_average,
        "Adjusted_Test_Loss": adjusted_average,
        "Total_Samples": total_samples,
        "Excluded_Samples": len(BAD_ROWS),
        "Remaining_Samples": adjusted_samples,
        "Excluded_Loss_Total": excluded_loss,
        "Excluded_Loss_Percentage": excluded_percentage
    }

    return comparison_result, excluded_details


# ============================================================
# EVALUATE ALL THREE MODELS
# ============================================================

all_results = []
all_excluded_details = []

for model_name, checkpoint in MODELS.items():

    result, details = evaluate_model(
        model_name,
        checkpoint
    )

    all_results.append(result)
    all_excluded_details.extend(details)

# ============================================================
# SAVE COMPARISON CSV
# ============================================================

comparison_df = pd.DataFrame(all_results)

comparison_df.to_csv(
    COMPARISON_FILE,
    index=False
)

# ============================================================
# SAVE INDIVIDUAL EXCLUDED SAMPLE DETAILS
# ============================================================

details_df = pd.DataFrame(all_excluded_details)

details_df.to_csv(
    EXCLUDED_DETAILS_FILE,
    index=False
)

# ============================================================
# FINAL RESULTS
# ============================================================

print("\n" + "=" * 80)
print("ORIGINAL VS ADJUSTED TEST LOSS COMPARISON")
print("=" * 80)

print(
    comparison_df[
        [
            "Model",
            "Original_Test_Loss",
            "Adjusted_Test_Loss",
            "Excluded_Samples"
        ]
    ].to_string(index=False)
)

print("\nExcluded test row indices:", BAD_ROWS)

print("\nGenerated files:")
print(COMPARISON_FILE)
print(EXCLUDED_DETAILS_FILE)

print("\nDiagnostic analysis completed successfully!")