import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

csv_file = Path("noniid_fedprox_metrics_100.csv")
plots_folder = Path("plots")
plots_folder.mkdir(exist_ok=True)

df = pd.read_csv(csv_file).sort_values("Round")

# Validation loss
plt.figure(figsize=(10, 5))
plt.plot(df["Round"], df["Loss"], label="Non-IID FedProx")
plt.xlabel("Round")
plt.ylabel("Validation Loss")
plt.title("Non-IID FedProx: Validation Loss")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(plots_folder / "noniid_fedprox_validation_loss_100.png", dpi=300)
plt.close()

# Accuracy, precision, recall, and F1
plt.figure(figsize=(10, 5))
for metric in ["Accuracy", "Precision", "Recall", "F1"]:
    plt.plot(df["Round"], df[metric], label=metric)

plt.xlabel("Round")
plt.ylabel("Score")
plt.title("Non-IID FedProx: Validation Metrics")
plt.ylim(0, 1)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(plots_folder / "noniid_fedprox_validation_metrics_100.png", dpi=300)
plt.close()

print(f"Graphs saved in: {plots_folder.resolve()}")