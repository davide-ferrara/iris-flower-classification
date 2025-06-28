import unittest
import numpy as np
from NeuralNetwork import NeuralNetwork
import json


error_loss = []
confidences = []


class TestPrime(unittest.TestCase):
    def setUp(self):
        np.set_printoptions(suppress=True, precision=5)

        self.dataset_path = "dataset/test_set.data"
        self.nn = NeuralNetwork.load("model.pkl")

        self.X, _ = self.nn.parse_dataset(self.dataset_path, 4)

        # Classe per Iris-Setosa
        self.class_zero_y = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])

        # Classe per Iris-Versicolor
        self.class_one_y = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]])

        # Classe per Iris_Virginica
        self.class_two_y = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]])

        self.iris_setosa = self.X[:9]
        self.iris_versicolor = self.X[10:20]
        self.iris_virginica = self.X[21:]

    def prototype_test(self, X, hot_vector, class_num):
        local_confidence = []
        for idx, x in enumerate(X):
            prob = self.nn.predict([x])
            prob = prob.flatten()

            pred = np.argmax(prob)
            confidence = np.max(prob) * 100

            error = self.nn.mean_squared_error(prob, hot_vector)
            error_loss.append(error)
            local_confidence.append(confidence)

            print(
                f"[{idx}] Prediction: {pred}, Class: {pred}, Confidence: {confidence:.2f}%, Error: {error}"
            )
            self.assertEqual(pred, class_num)

        confidences.append(np.mean(local_confidence))

    # Test per Iris-Setosa
    def test_one(self):
        self.prototype_test(self.iris_setosa, self.class_zero_y, 0)

    # Test per Iris-Versicolor
    def test_two(self):
        self.prototype_test(self.iris_versicolor, self.class_one_y, 1)

    # Test per Iris_Virginica
    def test_tree(self):
        self.prototype_test(self.iris_virginica, self.class_two_y, 2)

    @classmethod
    def tearDownClass(cls):
        # Salva le confidenze in un file txt semplice
        with open("confidences.csv", "w") as f:
            for conf in confidences:
                f.write(f"{conf},\n")

        print(f"Confidenze salvate: {confidences}")
        print("File confidences.txt creato!")


if __name__ == "__main__":
    unittest.main()
