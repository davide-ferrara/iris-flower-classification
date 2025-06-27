from NeuralNetwork import NeuralNetwork

if __name__ == "__main__":
    print(64 * "==")
    print(
        "Multilayer Perceptron per la classificazione del fiore Iris\nTraining della rete neurale in corso..."
    )

    dataset_path = "dataset/iris.data"
    test_dataset_path = "dataset/test_set.data"
    num_features = 4
    num_hidden = 32
    num_outputs = 3
    epoch = 1000
    learning_rate = 0.001
    error_threshold = 1e-5
    momentum = 0.9

    nn = NeuralNetwork(
        num_hidden,
        num_outputs,
        dataset_path,
        num_features,
        epoch,
        learning_rate,
        error_threshold,
        momentum,
        test_dataset_path,
    )

    nn.train()
    nn.save()

    print(64 * "==")
