import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

project_folder = Path(__file__).parent
plots_folder = project_folder / "plots"
plots_folder.mkdir(exist_ok=True)

# Add each experiment's CSV and graph filename prefix here
experiments = {
    "IID FedAvg - 50 Rounds": "iid_fedavg_metrics.csv",
    "IID FedAvg - 100 Rounds": "iid_fedavg_metrics_100.csv",
}

for experiment_name, csv_name in experiments.items():
    csv_file = project_folder / csv_name

    if not csv_file.exists():
        print(f"Skipping {experiment_name}: {csv_name} not found.")
        continue

    df = pd.read_csv(csv_file)
    df = df.drop_duplicates(subset="round", keep="last").sort_values("round")

    # Make filenames safe and consistent
    filename_prefix = csv_file.stem

    # Graph 1: Accuracy, Precision, Recall, and F1
    fig, ax = plt.subplots(figsize=(10, 6))

    for metric in ["accuracy", "precision", "recall", "f1"]:
        ax.plot(
            df["round"],
            df[metric],
            marker=".",
            markersize=3,
            label=metric.capitalize()
        )

    ax.set_xlabel("Federated Round")
    ax.set_ylabel("Score")
    ax.set_title(f"{experiment_name} - Validation Metrics")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    metrics_path = plots_folder / f"{filename_prefix}_metrics.png"
    fig.savefig(metrics_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    # Graph 2: Validation Loss
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df["round"], df["validation_loss"], marker=".", markersize=3)

    ax.set_xlabel("Federated Round")
    ax.set_ylabel("Validation Loss")
    ax.set_title(f"{experiment_name} - Validation Loss")
    ax.grid(True)
    fig.tight_layout()

    loss_path = plots_folder / f"{filename_prefix}_validation_loss.png"
    fig.savefig(loss_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print(f"Saved: {metrics_path}")
    print(f"Saved: {loss_path}")