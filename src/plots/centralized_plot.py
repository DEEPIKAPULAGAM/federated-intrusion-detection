import os
import matplotlib.pyplot as plt

os.makedirs("plots", exist_ok=True)

metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
centralized_values = [98.84, 87.03, 89.34, 88.17]

plt.figure(figsize=(8, 5))
plt.bar(metrics, centralized_values)

plt.title("Centralized Model: Validation Metrics")
plt.xlabel("Evaluation Metrics")
plt.ylabel("Score (%)")
plt.ylim(0, 100)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig("plots/centralized_metrics.png", dpi=300)
plt.close()

print("Centralized graph saved to plots/centralized_metrics.png")