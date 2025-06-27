import unittest
import numpy as np
from NeuralNetwork import NeuralNetwork


error_loss = []


class TestPrime(unittest.TestCase):
    def setUp(self):
        np.set_printoptions(suppress=True, precision=5)

        self.dataset_path = "dataset/test_set.data"
        self.nn = NeuralNetwork.load("model.pkl")

        self.X, _ = self.nn.parse_dataset(self.dataset_path, 4)

        self.class_zero_y = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        self.class_one_y = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]])
        self.class_two_y = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]])

        self.iris_setosa = self.X[:9]
        self.iris_versicolor = self.X[10:20]
        self.iris_virginica = self.X[21:]

    def prototype_test(self, X, hot_vector, class_num):
        for x in X:
            prob = self.nn.predict([x])
            prob = prob.flatten()

            pred = np.argmax(prob)
            confidenza = np.max(prob) * 100

            # error = hot_vector - prob.reshape(-1, 1)
            error = self.nn.mean_squared_error(prob, hot_vector)
            error_loss.append(error)

            print(
                f"Prediction: {pred}, Class: {pred}, Confidence: {confidenza:.2f}%, Error: {error}"
            )
            self.assertEqual(pred, class_num)

        print(error_loss)

    # Test per Iris-Setosa
    def test_one(self):
        self.prototype_test(self.iris_setosa, self.class_zero_y, 0)

    # Test per Iris-Versicolor
    def test_two(self):
        self.prototype_test(self.iris_versicolor, self.class_one_y, 1)

    # Test per Iris_Virginica
    def test_tree(self):
        self.prototype_test(self.iris_virginica, self.class_two_y, 2)


if __name__ == "__main__":
    unittest.main()
