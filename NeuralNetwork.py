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
    ):
        self.num_features = num_features
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.epochs = epochs
        self.learing_rate = learning_rate
        self.error_threshold = error_threshold
        self.momentum = momentum
        self.train_loss = []

        self.X, self.Y = self.parse_dataset(dataset_path, self.num_features)
        self.targets = self.one_hot()

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
    def one_hot(self):
        classes, inverse = np.unique(self.Y, return_inverse=True)
        one_hot = np.zeros((self.Y.shape[0], classes.size))
        one_hot[np.arange(self.Y.shape[0]), inverse] = 1
        return one_hot

    def init_weights(self, row, col):
        return np.random.uniform(-1, 1, (row, col))

    def init_bias(self, row):
        return np.zeros((row, 1))

    def feed_forward(self, input):
        # NOTE: Il prodotto matriciale np.dot(A, B) richiede che il numero di colonne di A corrisponda al numero di righe di B.
        # Per questo x è trasposta!
        self.hidden = np.dot(self.weights_ih, input) + self.bias_h
        self.hidden = self.sigmoid(self.hidden)

        self.outputs = np.dot(self.weights_ho, self.hidden) + self.bias_o
        self.outputs = self.sigmoid(self.outputs)

        return self.outputs

    def train(self, args=""):
        for epoch in range(self.epochs):
            a = self.feed_forward(self.X.T)
            y = self.targets.T
            error = self.mean_squared_error(a, y)
            self.train_loss.append(error)

            self.backprop(a, y)

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
        # 1. Calcola errore output
        error_output = a - y

        # 2. Delta output
        delta_output = error_output * (a * (1 - a))

        # 3. Aggiorna pesi e bias output layer con momentum
        grad_w_ho = np.dot(delta_output, self.hidden.T)
        grad_b_o = np.sum(delta_output, axis=1, keepdims=True)

        self.vel_w_ho = self.momentum * self.vel_w_ho - self.learing_rate * grad_w_ho
        self.vel_b_o = self.momentum * self.vel_b_o - self.learing_rate * grad_b_o

        self.weights_ho += self.vel_w_ho
        self.bias_o += self.vel_b_o

        # 4. Propaga errore a hidden layer
        error_hidden = np.dot(self.weights_ho.T, delta_output)
        delta_hidden = error_hidden * (self.hidden * (1 - self.hidden))

        # 5. Aggiorna pesi e bias hidden layer con momentum
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

    def save(self, model_name="model.pkl"):
        with open(model_name, "wb") as f:
            print(f"Model as been saved as: {model_name}!")
            pickle.dump(self, f)

    def get_epochs(self):
        return self.epochs

    def get_train_loss(self):
        return self.train_loss

    @staticmethod
    def load(model_path):
        with open(model_path, "rb") as f:
            nn = pickle.load(f)
        return nn
