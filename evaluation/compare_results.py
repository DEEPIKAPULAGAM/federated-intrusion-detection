from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# Get the folder where this script is saved
BASE_DIR = Path(__file__).resolve().parent

# Folder where the generated graphs will be saved
OUTPUT_DIR = BASE_DIR / "plots"
OUTPUT_DIR.mkdir(exist_ok=True)


# Metrics CSV files for the three experiments
# The IID file uses lowercase column names, so its columns are renamed below
files = {
    "IID FedAvg": {
        "filename": "iid_fedavg_metrics_100.csv",
        "columns": {
            "round": "Round",
            "validation_loss": "Loss",
            "accuracy": "Accuracy",
            "precision": "Precision",
            "recall": "Recall",
            "f1": "F1",
        },
    },
    "Non-IID FedAvg": {
        "filename": "noniid_fedavg_metrics_100.csv",
        "columns": {},
    },
    "Non-IID FedProx": {
        "filename": "noniid_fedprox_metrics_100.csv",
        "columns": {},
    },
}


# Store each experiment's metrics DataFrame here
data = {}

# Load the CSV files and make their column names consistent
for algorithm, settings in files.items():

    # Build the full path to the CSV file
    file_path = BASE_DIR / settings["filename"]

    # Stop with a clear error if a file is missing
    if not file_path.exists():
        raise FileNotFoundError(f"Missing metrics CSV: {file_path}")

    # Read the metrics CSV
    df = pd.read_csv(file_path)

    # Rename the IID CSV columns to match the other files
    if settings["columns"]:
        df = df.rename(columns=settings["columns"])

    # Sort metrics by communication round
    df = df.sort_values("Round")

    # Save the DataFrame using the experiment name
    data[algorithm] = df


# Confirm that all files contain the columns needed for plotting
required_columns = {
    "Round",
    "Loss",
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
}

for algorithm, df in data.items():
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"{algorithm} CSV is missing these columns: {missing}"
        )


# --------------------------------------------------
# Graph 1: Accuracy comparison
# --------------------------------------------------

plt.figure(figsize=(10, 5))

# Plot accuracy across communication rounds for each experiment
for algorithm, df in data.items():
    plt.plot(df["Round"], df["Accuracy"], label=algorithm)

plt.xlabel("Communication Round")
plt.ylabel("Accuracy")
plt.title("Accuracy Comparison Across Communication Rounds")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# Save the accuracy graph
plt.savefig(OUTPUT_DIR / "accuracy_comparison_100.png", dpi=300)
plt.close()


# --------------------------------------------------
# Graph 2: F1-score comparison
# --------------------------------------------------

plt.figure(figsize=(10, 5))

# Plot F1-score across communication rounds for each experiment
for algorithm, df in data.items():
    plt.plot(df["Round"], df["F1"], label=algorithm)

plt.xlabel("Communication Round")
plt.ylabel("F1 Score")
plt.title("F1 Score Comparison Across Communication Rounds")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# Save the F1-score graph
plt.savefig(OUTPUT_DIR / "f1_comparison_100.png", dpi=300)
plt.close()


# --------------------------------------------------
# Graph 3: Validation loss comparison
# --------------------------------------------------

plt.figure(figsize=(10, 5))

# Plot validation loss across communication rounds
for algorithm, df in data.items():
    plt.plot(df["Round"], df["Loss"], label=algorithm)

plt.xlabel("Communication Round")
plt.ylabel("Validation Loss")
plt.title("Validation Loss Comparison Across Communication Rounds")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# Save the loss graph
plt.savefig(OUTPUT_DIR / "loss_comparison_100.png", dpi=300)
plt.close()


# --------------------------------------------------
# Graph 4: Final metrics comparison
# --------------------------------------------------

# Metrics to compare in the final bar chart
metrics = ["Accuracy", "Precision", "Recall", "F1"]

# Store each experiment's final-round metric values
final_values = {}

for algorithm, df in data.items():

    # Get the row with the highest communication round
    final_row = df.iloc[-1]

    # Extract the final values for the selected metrics
    final_values[algorithm] = [
        final_row[metric] for metric in metrics
    ]


# Set up bar positions and width
x = list(range(len(metrics)))
width = 0.25
algorithms = list(data.keys())

plt.figure(figsize=(11, 5))

# Draw a set of bars for each experiment
for index, algorithm in enumerate(algorithms):

    # Shift the bars so each experiment appears side by side
    positions = [value + (index - 1) * width for value in x]

    plt.bar(
        positions,
        final_values[algorithm],
        width,
        label=algorithm,
    )

plt.xlabel("Evaluation Metric")
plt.ylabel("Score")
plt.title("Final Metrics Comparison")
plt.xticks(x, metrics)
plt.grid(axis="y", alpha=0.3)
plt.legend()
plt.tight_layout()

# Save the final metrics graph
plt.savefig(OUTPUT_DIR / "final_metrics_comparison_100.png", dpi=300)
plt.close()


# --------------------------------------------------
# Print the final metric values
# --------------------------------------------------

print("Performance comparison graphs saved in:", OUTPUT_DIR)

print("\nFinal metrics:")
for algorithm, values in final_values.items():
    print(f"\n{algorithm}:")

    for metric, value in zip(metrics, values):
        print(f"  {metric}: {value:.6f}")