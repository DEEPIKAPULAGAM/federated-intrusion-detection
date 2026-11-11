# Mitigating Model Drift in Federated Learning-Based Network Intrusion Detection under Non-IID Traffic Skew

## Project Overview

This project investigates Federated Learning for Network Intrusion Detection using the UNSW-NB15 dataset.

It compares the performance of Federated Averaging (FedAvg) and Federated Proximal (FedProx) algorithms under IID and Non-IID data distributions.

The primary objective is to study client drift and evaluate how FedProx helps mitigate model drift under heterogeneous network traffic distributions.

## Project Objectives

- Implement Federated Learning for network intrusion detection.
- Compare IID and Non-IID data distributions.
- Evaluate FedAvg and FedProx algorithms.
- Analyze client drift across federated clients.
- Compare model performance using classification metrics.

## Technologies Used

- Python
- PyTorch
- Flower
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## Dataset

**UNSW-NB15**

The dataset contains normal network traffic and various attack categories.

The project uses numerical network traffic features and PCA-transformed components for binary classification:

- 0: Normal
- 1: Attack

Dataset source: https://research.unsw.edu.au/projects/unsw-nb15-dataset

## Experimental Setup

Three main experiments were conducted:

1. IID FedAvg
2. Non-IID FedAvg
3. Non-IID FedProx

The federated setup uses three simulated clients, 100 communication rounds and three local epochs.

FedProx uses a proximal regularization parameter of μ = 0.01.

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- Binary Cross-Entropy Loss
- Confusion Matrix
- Client Drift

## Repository Structure

- `src/`: Model, client, server and training implementation.
- `evaluation/`: Testing and analysis scripts.
- `results/`: Experimental metrics and drift analysis.
- `plots/`: Training, dataset, drift and testing visualizations.
- `report/`: Final academic project report.

## Key Findings

- IID FedAvg achieved the highest test accuracy and F1-score.
- Non-IID FedAvg achieved the highest precision.
- Non-IID FedProx achieved the highest recall.
- FedProx reduced average client drift by approximately 73.16% compared with Non-IID FedAvg.

## How to Run

1. Clone the repository.
2. Install the required dependencies.
3. Download the UNSW-NB15 dataset.
4. Prepare the IID and Non-IID client datasets using the project preprocessing pipeline.
5. Run the training scripts.
6. Run the evaluation scripts to compare model performance.

Detailed execution instructions will be provided with the final project documentation.

## Note

The dataset and trained model checkpoints are not included in this repository. They must be prepared or obtained separately before running the complete experiments.

## Project

Federated Learning-Based Network Intrusion Detection under Non-IID Traffic Skew.