import numpy as np
import matplotlib.pyplot as plt
from NeuralNetwork import NeuralNetwork

dataset_path = "dataset/test_set.data"
nn = NeuralNetwork.load("model.pkl")

train_loss_x = np.arange(0, nn.get_epochs())
train_loss_y = nn.get_train_loss()

test_loss_x = train_loss_x
test_loss_y = nn.get_test_loss()
# Creare il plot
plt.plot(train_loss_x, train_loss_y, label="Train Loss", color="blue")
plt.plot(test_loss_x, test_loss_y, label="Test Loss", color="orange")

plt.title("Errore totale su training e test")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.axhline(0, color="black", linewidth=0.5, ls="--")
plt.axvline(0, color="black", linewidth=0.5, ls="--")
plt.grid()
plt.legend()
plt.show()
