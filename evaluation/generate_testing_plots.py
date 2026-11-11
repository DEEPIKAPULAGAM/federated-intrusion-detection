
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)

# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

PREDICTION_FILES = {
    "IID FedAvg": "test_predictions/iid_fedavg_predictions.csv",
    "Non-IID FedAvg": "test_predictions/non-iid_fedavg_predictions.csv",
    "Non-IID FedProx": "test_predictions/non-iid_fedprox_predictions.csv"
}

OUTPUT_DIR = "plots/testing_analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------
# 2. LOAD PREDICTIONS
# --------------------------------------------------

models = {}

for model_name, file_path in PREDICTION_FILES.items():

    df = pd.read_csv(file_path)

    print(f"\n{model_name}")
    print("Columns:", df.columns.tolist())

    required = ["Actual_Label", "Probability"]

    if not all(col in df.columns for col in required):
        raise ValueError(
            f"{file_path} must contain columns: {required}. "
            f"Found: {df.columns.tolist()}"
        )

    y_true = df["Actual_Label"].to_numpy(dtype=int)
    y_prob = df["Probability"].to_numpy(dtype=float)

    if not np.isfinite(y_prob).all():
        raise ValueError(f"Invalid probabilities in {file_path}")

    if ((y_prob < 0) | (y_prob > 1)).any():
        raise ValueError(f"Probabilities outside [0, 1] in {file_path}")

    y_pred = (y_prob >= 0.5).astype(int)

    models[model_name] = {
        "actual": y_true,
        "probability": y_prob,
        "predicted": y_pred
    }

# Check that all models use the same test labels
first_labels = next(iter(models.values()))["actual"]

for name, result in models.items():
    if not np.array_equal(first_labels, result["actual"]):
        raise ValueError(
            f"{name} does not have the same test labels/order."
        )

print("\nAll prediction files loaded successfully.")

# --------------------------------------------------
# 3. CALCULATE TEST METRICS
# --------------------------------------------------

metrics = []

for name, result in models.items():

    y_true = result["actual"]
    y_pred = result["predicted"]

    metrics.append({
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "F1": f1_score(
            y_true, y_pred, zero_division=0
        )
    })

metrics_df = pd.DataFrame(metrics)

metrics_df.to_csv(
    f"{OUTPUT_DIR}/test_metrics_comparison.csv",
    index=False
)

# --------------------------------------------------
# 4. PLOT 1: TEST METRICS COMPARISON
# --------------------------------------------------

plot_metrics = metrics_df.set_index("Model")

ax = plot_metrics.plot(
    kind="bar",
    figsize=(11, 6),
    ylim=(0, 1.05),
    width=0.75
)

ax.set_title("Test Performance Comparison")
ax.set_ylabel("Score")
ax.set_xlabel("Federated Learning Models")
ax.legend(title="Metrics")
ax.grid(axis="y", alpha=0.3)

plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(
    f"{OUTPUT_DIR}/test_metrics_comparison.png",
    dpi=300
)
plt.close()

# --------------------------------------------------
# 5. PLOTS 2-4: CONFUSION MATRICES
# --------------------------------------------------

for name, result in models.items():

    cm = confusion_matrix(
        result["actual"],
        result["predicted"],
        labels=[0, 1]
    )

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "Attack"],
        yticklabels=["Normal", "Attack"]
    )

    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")

    plt.tight_layout()

    filename = name.lower().replace(" ", "_").replace("-", "")

    plt.savefig(
        f"{OUTPUT_DIR}/{filename}_confusion_matrix.png",
        dpi=300
    )

    plt.close()

# --------------------------------------------------
# 6. PLOT 5: TEST LOSS COMPARISON
# --------------------------------------------------

# Use the official logits-based test losses.
# Do not recalculate BCE from rounded probabilities.

official_losses = {
    "IID FedAvg": 0.144402,
    "Non-IID FedAvg": 1.918761,
    "Non-IID FedProx": 0.024759
}

plt.figure(figsize=(9, 5))

bars = plt.bar(
    list(official_losses.keys()),
    list(official_losses.values())
)

plt.title("Test Loss Comparison")
plt.ylabel("Binary Cross-Entropy Loss")
plt.xlabel("Models")
plt.grid(axis="y", alpha=0.3)

for bar, value in zip(bars, official_losses.values()):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.6f}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.savefig(
    f"{OUTPUT_DIR}/test_loss_comparison.png",
    dpi=300
)
plt.close()


# --------------------------------------------------
# 6.1 PLOT: ORIGINAL VS ADJUSTED TEST LOSS
# --------------------------------------------------

original_losses = {
    "IID FedAvg": 0.144402,
    "Non-IID FedAvg": 1.918761,
    "Non-IID FedProx": 0.024759
}

adjusted_losses = {
    "IID FedAvg": 0.019747,
    "Non-IID FedAvg": 0.023767,
    "Non-IID FedProx": 0.023904
}

model_names = list(original_losses.keys())

x = np.arange(len(model_names))
width = 0.35

plt.figure(figsize=(10, 6))

bars1 = plt.bar(
    x - width / 2,
    list(original_losses.values()),
    width,
    label="Original Test Loss"
)

bars2 = plt.bar(
    x + width / 2,
    list(adjusted_losses.values()),
    width,
    label="Adjusted Test Loss"
)

plt.title("Original vs Adjusted Test Loss Comparison")
plt.xlabel("Federated Learning Models")
plt.ylabel("Binary Cross-Entropy Loss")
plt.xticks(x, model_names)
plt.legend()
plt.grid(axis="y", alpha=0.3)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.4f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/original_vs_adjusted_loss.png",
    dpi=300
)

plt.close()

print("Original vs adjusted test loss graph generated.")


# --------------------------------------------------
# 7. PLOT 6: ROC CURVES
# --------------------------------------------------

plt.figure(figsize=(8, 6))

for name, result in models.items():

    fpr, tpr, _ = roc_curve(
        result["actual"],
        result["probability"]
    )

    roc_auc = auc(fpr, tpr)

    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC = {roc_auc:.4f})"
    )

plt.plot([0, 1], [0, 1], linestyle="--")

plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(
    f"{OUTPUT_DIR}/roc_curve_comparison.png",
    dpi=300
)
plt.close()

# --------------------------------------------------
# 8. PLOT 7: PRECISION-RECALL CURVES
# --------------------------------------------------

plt.figure(figsize=(8, 6))

for name, result in models.items():

    precision, recall, _ = precision_recall_curve(
        result["actual"],
        result["probability"]
    )

    ap = average_precision_score(
        result["actual"],
        result["probability"]
    )

    plt.plot(
        recall,
        precision,
        label=f"{name} (AP = {ap:.4f})"
    )

plt.title("Precision-Recall Curve Comparison")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(
    f"{OUTPUT_DIR}/precision_recall_comparison.png",
    dpi=300
)
plt.close()

# --------------------------------------------------
# 9. PRINT FINAL METRICS
# --------------------------------------------------

print("\nFINAL TEST METRICS")
print(metrics_df.to_string(index=False))

print("\nAll testing plots saved in:")
print(OUTPUT_DIR)
