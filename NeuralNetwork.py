import numpy as np
import pandas as pd
import pickle


class NeuralNetwork:
    def __init__(
        self,
        num_hidden,
        num_outputs,
        dataset_path,
        num_features,
        epochs,
        learning_rate,
        error_threshold,
        momentum,
        test_dataset_path=None,
    ):
        self.num_features = num_features
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.epochs = epochs
        self.learing_rate = learning_rate
        self.error_threshold = error_threshold
        self.momentum = momentum
        self.train_loss = []
        self.test_loss = []
        self.test_dataset_path = test_dataset_path

        self.X, self.Y = self.parse_dataset(dataset_path, self.num_features)
        self.targets = self.one_hot(self.Y)

        if self.test_dataset_path is not None:
            self.X_test, self.Y_test = self.parse_dataset(
                self.test_dataset_path, self.num_features
            )
            self.targets_test = self.one_hot(self.Y_test)

        # Pesi da input layer verso hidden layer
        self.weights_ih = self.init_weights(self.num_hidden, self.num_features)
        self.weights_ho = self.init_weights(self.num_outputs, self.num_hidden)

        self.bias_h = self.init_bias(self.num_hidden)
        self.bias_o = self.init_bias(self.num_outputs)

        self.vel_w_ho = np.zeros_like(self.weights_ho)
        self.vel_b_o = np.zeros_like(self.bias_o)
        self.vel_w_ih = np.zeros_like(self.weights_ih)
        self.vel_b_h = np.zeros_like(self.bias_h)

    @staticmethod
    def parse_dataset(dataset_path, num_features):
        df = pd.read_csv(dataset_path, header=None)
        X = np.array(df.iloc[:, :num_features].values)
        Y = np.array(df.iloc[:, num_features].values)
        return X, Y

    # https://wandb.ai/mostafaibrahim17/ml-articles/reports/One-Hot-Encoding-Creating-a-NumPy-Array-Using-Weights-Biases--Vmlldzo2MzQzNTQ5
    def one_hot(self, Y):
        classes, inverse = np.unique(Y, return_inverse=True)
        one_hot = np.zeros((Y.shape[0], classes.size))
        one_hot[np.arange(Y.shape[0]), inverse] = 1
        return one_hot

    def init_weights(self, row, col):
        return np.random.uniform(-1, 1, (row, col))

    def init_bias(self, row):
        return np.zeros((row, 1))

    def feed_forward(self, input):
        # NOTE: Il prodotto matriciale np.dot(A, B) richiede che il numero di colonne di A corrisponda al numero di righe di B.
        # Per questo x è trasposta!
        self.hidden = np.dot(self.weights_ih, input) + self.bias_h
        self.hidden = self.relu(self.hidden)

        self.outputs = np.dot(self.weights_ho, self.hidden) + self.bias_o
        self.outputs = self.softmax(self.outputs)

        return self.outputs

    def train(self, args=""):
        for epoch in range(self.epochs):
            a = self.feed_forward(self.X.T)
            y = self.targets.T

            self.backprop(a, y)

            # Errore con dati di training
            error = self.mean_squared_error(a, y)
            self.train_loss.append(error)

            if self.test_dataset_path is not None:
                a_test = self.feed_forward(self.X_test.T)
                y_test = self.targets_test.T

                # Errore con dati di test
                error_test = self.mean_squared_error(a_test, y_test)
                self.test_loss.append(error_test)

            if "-v" in args:
                print(f"Epoch {epoch + 1}, error: {error}")

            if epoch == self.epochs - 1:
                print(f"Error: {error}")

            if error <= self.error_threshold:
                print(
                    f"Training stopped at epoch {epoch + 1}, error threshold reached."
                )
                break

    def predict(self, input):
        input = np.array(input)
        prediction = self.feed_forward(input.T)
        return prediction

    # Il Momentum aiuta a velocizzare la discesa del gradiente e a evitare minimi locali.
    def backprop(self, a, y):
        # Calcolo errore output
        error_output = a - y

        # Delta output
        delta_output = error_output

        # Aggiorno pesi e bias output layer con momentum
        grad_w_ho = np.dot(delta_output, self.hidden.T)
        grad_b_o = np.sum(delta_output, axis=1, keepdims=True)

        self.vel_w_ho = self.momentum * self.vel_w_ho - self.learing_rate * grad_w_ho
        self.vel_b_o = self.momentum * self.vel_b_o - self.learing_rate * grad_b_o

        self.weights_ho += self.vel_w_ho
        self.bias_o += self.vel_b_o

        # Propago errore a hidden layer
        error_hidden = np.dot(self.weights_ho.T, delta_output)
        delta_hidden = error_hidden * self.relu(self.hidden, True)  # Derivata

        # Aggiorno pesi e bias hidden layer con momentum
        grad_w_ih = np.dot(delta_hidden, self.X)
        grad_b_h = np.sum(delta_hidden, axis=1, keepdims=True)

        self.vel_w_ih = self.momentum * self.vel_w_ih - self.learing_rate * grad_w_ih
        self.vel_b_h = self.momentum * self.vel_b_h - self.learing_rate * grad_b_h

        self.weights_ih += self.vel_w_ih
        self.bias_h += self.vel_b_h

    @staticmethod
    def mean_squared_error(a, y, derivate=False):
        a = np.array(a)
        y = np.array(y)

        if derivate:
            return 2 * (a - y) / a.size

        # c = (a - y)^2
        return np.mean(np.square(a - y))

    @staticmethod
    def softmax(x):
        x = x - np.max(x, axis=0, keepdims=True)
        e_x = np.exp(x)
        return e_x / np.sum(e_x, axis=0, keepdims=True)

    @staticmethod
    def sigmoid(x, derivative=False):
        s = 1 / (1 + np.exp(-x))
        if derivative:
            return s * (1 - s)
        return s

    @staticmethod
    def relu(x, derivative=False):
        if derivative:
            return (x > 0).astype(float)
        return np.maximum(0, x)

    @staticmethod
    def tanh(x, derivative=False):
        t = np.tanh(x)
        if derivative:
            return 1 - np.square(t)
        return t

    @staticmethod
    def load(model_path):
        with open(model_path, "rb") as f:
            nn = pickle.load(f)
        return nn

    def save(self, model_name="model.pkl"):
        with open(model_name, "wb") as f:
            print(f"Model as been saved as: {model_name}!")
            pickle.dump(self, f)

    def get_epochs(self):
        return self.epochs

    def get_train_loss(self):
        return self.train_loss

    def get_test_loss(self):
        return self.test_loss
