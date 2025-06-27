import numpy as np
import matplotlib.pyplot as plt
from NeuralNetwork import NeuralNetwork

dataset_path = "dataset/test_set.data"
nn = NeuralNetwork.load("model.pkl")

train_loss_x = np.arange(0, nn.get_epochs())
train_loss_y = nn.get_train_loss()

test_loss_x = np.arange(0, 28)
test_loss_y = [
    0.22127560331768845,
    0.22127638929469354,
    0.22124247226894914,
    0.22114763413064792,
    0.22115829609630153,
    0.2212206592792288,
    0.2212764440943087,
    0.22123357460831994,
    0.22128331376995009,
    0.44225013388525175,
    0.44437459294240106,
    0.44441144703815083,
    0.444403237927719,
    0.4443232924095671,
    0.44278127360358394,
    0.4442126331053599,
    0.4419830282337962,
    0.4439167461060025,
    0.440798448575085,
    0.444363304712983,
    0.44419236062416084,
    0.4437375030996512,
    0.4443485283731008,
    0.44419240618808786,
    0.4442568720470075,
    0.4441555769179566,
    0.4418671456747049,
    0.4442103089734406,
]

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
