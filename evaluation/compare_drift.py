from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# Get the folder where this Python script is saved
BASE_DIR = Path(__file__).resolve().parent

# Folder containing the six 100-round drift CSV files
DRIFT_DIR = BASE_DIR / "drift_analysis"

# Folder where the graphs and combined CSV files will be saved
OUTPUT_DIR = BASE_DIR / "plots"
OUTPUT_DIR.mkdir(exist_ok=True)


# CSV filename patterns for the two algorithms
# The {} is replaced by the client number: 1, 2, or 3
files = {
    "FedAvg": "drift_noniid_client_{}_100.csv",
    "FedProx": "drift_fedprox_client_{}_100.csv",
}


# Store the data from all six CSV files here
frames = []

# Load each client's drift data for both algorithms
for algorithm, filename_pattern in files.items():
    for client in range(1, 4):

        # Create the full path to this client's CSV file
        file_path = DRIFT_DIR / filename_pattern.format(client)

        # Stop with an error if the expected CSV file is missing
        if not file_path.exists():
            raise FileNotFoundError(f"Missing CSV file: {file_path}")

        # Read the CSV file
        df = pd.read_csv(file_path)

        # Add columns to identify the algorithm and client
        df["Algorithm"] = algorithm
        df["Client"] = client

        # Add this client's data to the list
        frames.append(df)


# Combine all six client CSVs into one DataFrame
drift = pd.concat(frames, ignore_index=True)


# Check that all required columns exist in the combined data
required_columns = {"Round", "Model_Drift", "Algorithm", "Client"}
missing = required_columns - set(drift.columns)

if missing:
    raise ValueError(f"Missing required CSV columns: {missing}")


# Calculate the average model drift across the three clients for each round
mean_drift = (
    drift.groupby(["Algorithm", "Round"], as_index=False)["Model_Drift"]
    .mean()
)


# Save the combined client data and average drift data as CSV files
drift.to_csv(OUTPUT_DIR / "combined_drift_100.csv", index=False)
mean_drift.to_csv(OUTPUT_DIR / "mean_drift_by_round_100.csv", index=False)


# --------------------------------------------------
# Graph 1: Average drift comparison
# --------------------------------------------------

plt.figure(figsize=(10, 5))

# Draw one line for each algorithm
for algorithm in ["FedAvg", "FedProx"]:
    data = mean_drift[mean_drift["Algorithm"] == algorithm]

    plt.plot(
        data["Round"],
        data["Model_Drift"],
        label=algorithm,
    )

# Add graph labels and formatting
plt.xlabel("Federated Round")
plt.ylabel("Average Model Drift")
plt.title("Non-IID FedAvg vs FedProx: Average Client Model Drift")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# Save the graph as a high-resolution image
plt.savefig(OUTPUT_DIR / "average_drift_comparison_100.png", dpi=300)
plt.close()


# --------------------------------------------------
# Graph 2: Client-wise drift comparison
# --------------------------------------------------

# Create three side-by-side plots, one for each client
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

for client, ax in zip([1, 2, 3], axes):

    # Select the data for the current client
    client_data = drift[drift["Client"] == client]

    # Plot FedAvg and FedProx drift for this client
    for algorithm in ["FedAvg", "FedProx"]:
        data = client_data[client_data["Algorithm"] == algorithm]

        ax.plot(
            data["Round"],
            data["Model_Drift"],
            label=algorithm,
        )

    # Add labels and formatting for this client's plot
    ax.set_title(f"Client {client}")
    ax.set_xlabel("Federated Round")
    ax.grid(True, alpha=0.3)

# Add a shared y-axis label and legend
axes[0].set_ylabel("Model Drift")
axes[-1].legend()

fig.suptitle("Client-wise Drift: Non-IID FedAvg vs FedProx")
fig.tight_layout()

# Save the client-wise graph
fig.savefig(OUTPUT_DIR / "client_wise_drift_comparison_100.png", dpi=300)
plt.close(fig)


# Print a completion message and average drift summary
print("Drift comparison graphs and CSVs saved in:", OUTPUT_DIR)
print("\nAverage drift across all rounds:")
print(mean_drift.groupby("Algorithm")["Model_Drift"].mean())