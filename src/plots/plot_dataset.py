
import pandas as pd
import matplotlib.pyplot as plt
import os

# CSV file locations
iid_files = {
    "Client 1": r"C:\Users\DELL\Downloads\client_1.csv",
    "Client 2": r"C:\Users\DELL\Downloads\client_2.csv",
    "Client 3": r"C:\Users\DELL\Downloads\client_3.csv"
}

non_iid_files = {
    "Client 1": r"C:\federated-ids\non_iid\noniidclient_1.csv",
    "Client 2": r"C:\federated-ids\non_iid\noniidclient_2.csv",
    "Client 3": r"C:\federated-ids\non_iid\noniidclient_3.csv"
}

# Folder to save plots
os.makedirs("plots", exist_ok=True)


# Read class counts without loading the entire CSV
def get_class_counts(file_path):
    counts = {0: 0, 1: 0}

    for chunk in pd.read_csv(
        file_path,
        chunksize=100000,
        usecols=["Label"]
    ):
        values = chunk["Label"].value_counts()

        for label in [0, 1]:
            counts[label] += int(values.get(label, 0))

    return counts


# Read attack category counts
def get_attack_counts(file_path):
    counts = {}

    for chunk in pd.read_csv(
        file_path,
        chunksize=100000,
        usecols=["attack_cat"],
        dtype={"attack_cat": "string"}
    ):
        values = chunk["attack_cat"].dropna().value_counts()

        for category, count in values.items():
            counts[category] = counts.get(category, 0) + int(count)

    return counts


# 1. Overall Normal vs Attack distribution
def plot_overall_distribution():

    iid_total = {0: 0, 1: 0}
    non_iid_total = {0: 0, 1: 0}

    for path in iid_files.values():
        counts = get_class_counts(path)
        for label in [0, 1]:
            iid_total[label] += counts[label]

    for path in non_iid_files.values():
        counts = get_class_counts(path)
        for label in [0, 1]:
            non_iid_total[label] += counts[label]

    labels = ["Normal", "Attack"]
    x = range(2)
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(
        [i - width/2 for i in x],
        [iid_total[0], iid_total[1]],
        width, label="IID"
    )
    plt.bar(
        [i + width/2 for i in x],
        [non_iid_total[0], non_iid_total[1]],
        width, label="Non-IID"
    )

    plt.xticks(list(x), labels)
    plt.ylabel("Number of Records")
    plt.title("Overall Normal vs Attack Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/overall_class_distribution.png", dpi=300)
    plt.show()


# 2. IID vs Non-IID client class distribution
def plot_client_distribution():

    for title, files in [
        ("IID", iid_files),
        ("Non-IID", non_iid_files)
    ]:

        normal_counts = []
        attack_counts = []

        for path in files.values():
            counts = get_class_counts(path)
            normal_counts.append(counts[0])
            attack_counts.append(counts[1])

        x = range(3)
        width = 0.35

        plt.figure(figsize=(8, 5))
        plt.bar(
            [i - width/2 for i in x],
            normal_counts, width, label="Normal"
        )
        plt.bar(
            [i + width/2 for i in x],
            attack_counts, width, label="Attack"
        )

        plt.xticks(list(x), list(files.keys()))
        plt.ylabel("Number of Records")
        plt.title(f"{title} Client Class Distribution")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            f"plots/{title.lower()}_client_distribution.png",
            dpi=300
        )
        plt.show()


# 3. Attack category distribution across clients
def plot_attack_categories():

    for title, files in [
        ("IID", iid_files),
        ("Non-IID", non_iid_files)
    ]:

        category_data = {}

        for client, path in files.items():
            category_data[client] = get_attack_counts(path)

        all_categories = sorted(
            set().union(
                *(counts.keys() for counts in category_data.values())
            )
        )

        x = range(len(all_categories))
        width = 0.25

        plt.figure(figsize=(13, 6))

        for index, (client, counts) in enumerate(category_data.items()):
            values = [counts.get(cat, 0) for cat in all_categories]

            positions = [
                i + (index - 1) * width for i in x
            ]

            plt.bar(positions, values, width, label=client)

        plt.xticks(list(x), all_categories, rotation=45, ha="right")
        plt.ylabel("Number of Attack Records")
        plt.title(f"{title} Attack Category Distribution")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            f"plots/{title.lower()}_attack_categories.png",
            dpi=300
        )
        plt.show()


# Run all plots
if __name__ == "__main__":

    print("Generating dataset distribution plots...")

    plot_overall_distribution()
    plot_client_distribution()
    plot_attack_categories()

    print("All plots saved successfully in the plots folder!")
