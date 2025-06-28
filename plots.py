import numpy as np
import csv
import matplotlib.pyplot as plt
from NeuralNetwork import NeuralNetwork


def error_loss_plot():
    train_loss_x = np.arange(0, nn.get_epochs())
    train_loss_y = nn.get_train_loss()

    test_loss_x = train_loss_x
    test_loss_y = nn.get_test_loss()

    plt.plot(train_loss_x, train_loss_y, label="Train Loss", color="blue")
    plt.plot(test_loss_x, test_loss_y, label="Test Loss", color="orange")

    plt.title("Errore totale su training e test")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.axhline(0, color="black", linewidth=0.5, ls="--")
    plt.axvline(0, color="black", linewidth=0.5, ls="--")
    plt.grid()
    plt.legend()
    plt.savefig("error_loss_plot.jpeg")


def confidence_candel_plot():
    tipologie = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

    with open("confidences.csv", newline="") as csvfile:
        data = list(csv.reader(csvfile))

    percentuali = [float(data[0][0]), float(data[1][0]), float(data[2][0])]
    print(percentuali)

    colori = ["skyblue", "green", "orchid"]

    _, ax = plt.subplots(figsize=(8, 6))

    bars = ax.bar(tipologie, percentuali, color=colori, width=0.6)

    ax.set_ylabel("Percentuale (%)", fontsize=12)
    ax.set_xlabel("Tipologia", fontsize=12)
    ax.set_ylim(0, 105)

    ax.grid(axis="y", alpha=0.3, linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)

    ax.set_yticks(np.arange(0, 101, 20))

    for bar in bars:
        bar.set_edgecolor("black")
        bar.set_linewidth(0.8)

    plt.tight_layout()

    plt.show()

    plt.savefig("confidence_candel_plot.jpeg", dpi=300, bbox_inches="tight")


dataset_path = "dataset/test_set.data"
nn = NeuralNetwork.load("model.pkl")

error_loss_plot()
confidence_candel_plot()
