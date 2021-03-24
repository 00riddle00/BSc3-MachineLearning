#!/usr/bin/env python3

import numpy as np

def run(neural_network, init_weights):
    neural_network.synaptic_weights = init_weights

    neural_network.train(given_train_inputs,
                        given_train_outputs,
                        learning_rate,
                        activation_function,
                        iterations,
                        epochs)

    train_outputs = neural_network.think(given_train_inputs, activation_function)
    test_outputs = neural_network.think(given_test_inputs, activation_function)

    if activation_function == 'sigmoid':
        train_outputs = np.array([
            1 if xi > 0.9 else 0 if xi < 0.1 else np.nan for xi in train_outputs])

        test_outputs = np.array([
            1 if xi > 0.9 else 0 if xi < 0.1 else np.nan for xi in test_outputs])

    train_results = np.array([sum(x) for x in zip(
        given_train_outputs.flatten(), train_outputs)])

    test_results = np.array([sum(x) for x in zip(
        given_test_outputs.flatten(), test_outputs)])

    unique, counts = np.unique(train_results[~np.isnan(train_results)], return_counts=True)
    train_results = dict(zip(unique, counts))

    unique, counts = np.unique(test_results[~np.isnan(test_results)], return_counts=True)
    test_results = dict(zip(unique, counts))

    train_accuracy = ( train_results.get(0.0, 0) + train_results.get(2.0, 0) ) / len(train_outputs)
    test_accuracy = ( test_results.get(0.0, 0) + test_results.get(2.0, 0) ) / len(test_outputs)

    print(f'{iterations},{train_accuracy:.2f},{test_accuracy:.2f}')


class NeuralNetwork():

    def __init__(self):
        np.random.seed(1)

        # 2x - 1, where x is random number in interval [0.0,1.0),
        # so the values will be in range [-1,1) with equal probability
        self.synaptic_weights = 2 * np.random.random((5, 1)) - 1

    def threshold_fn(self, x):
        return np.heaviside(x, 0)

    def sigmoid_fn(self, x):
        sig = 1 / (1 + np.exp(-x))     # Define sigmoid function
        sig = np.minimum(sig, 0.9999)  # Set upper bound
        sig = np.maximum(sig, 0.0001)  # Set lower bound
        return sig

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self,
              train_inputs,
              train_outputs,
              learning_rate,
              activation_function,
              train_iterations,
              train_epochs):

        for epoch in range(train_epochs):
            for iteration in range(train_iterations):
                output = self.think(train_inputs, activation_function)
                error = train_outputs - output

                adjustments = np.dot(train_inputs.T, error * learning_rate)
                self.synaptic_weights = self.synaptic_weights + adjustments

    def think(self, inputs, activation_function):
        inputs = inputs.astype(float)
        if (activation_function == 'threshold'):
            output = self.threshold_fn(np.dot(inputs, self.synaptic_weights))
        elif (activation_function == 'sigmoid'):
            output = self.sigmoid_fn(np.dot(inputs, self.synaptic_weights))
        else:
            raise ValueError('Incorrect activation function specified')
            exit(1)

        return output

if __name__ == '__main__':
    # =================================================================
    # changeable parameters
    # =================================================================

    # classification task is one of {1,2}
    task = 1
    # learning_rate = 0

    # activation function is one of {'threshold', 'sigmoid'}
    activation_function = 'threshold'
    iterations = 0
    # epochs = 0

    # =================================================================
    # Reading from a file
    # =================================================================

    data = np.genfromtxt('dataset/iris.data', delimiter=',')
    data[:, 4] = 1

    data_setosa = data[:50]
    data_versicolor = data[50:100]
    data_virginica = data[100:150]
    data_versicolor_and_virginica = data[50:150]

    data_train_setosa = data_setosa[:45]
    data_test_setosa = data_setosa[45:50]

    data_train_versicolor = data_versicolor[:45]
    data_test_versicolor = data_versicolor[45:50]

    data_train_virginica = data_virginica[:45]
    data_test_virginica = data_virginica[45:50]

    data_train_versicolor_and_virginica = np.concatenate((
        data_train_versicolor, data_train_virginica))

    data_test_versicolor_and_virginica = np.concatenate((
        data_test_versicolor, data_test_virginica))

    # =================================================================
    # Setting training & test data
    # =================================================================

    if (task == 1):

        if iterations > 135:
            raise ValueError('Maximum iteration count in Task 1 is 135')
            exit(1)

        # -------------------------------------------------------------
        # Classification task no.1 (Setosa vs. Versicolor_&_Virginica)
        # -------------------------------------------------------------

        # -------------------- Training data --------------------------

        given_train_inputs = []

        # mix the data (every second element from another class)
        for element in zip(data_train_setosa, data_train_versicolor_and_virginica[:45]):
            given_train_inputs.extend(element)

        given_train_inputs = np.concatenate((np.array(given_train_inputs), data_train_versicolor_and_virginica[45:]))

        # classes should correspond to the input elements
        given_train_outputs = np.zeros(90)

        # every second element is of class 1
        given_train_outputs[1::2] = 1
        given_train_outputs = np.concatenate((given_train_outputs, np.ones(45))).reshape(1, -1).T

        # -------------------- Testing data --------------------------

        given_test_inputs = np.concatenate((
            data_test_setosa, data_test_versicolor_and_virginica))

        given_test_outputs = np.concatenate((
            np.zeros(5), np.ones(10))).reshape(1, -1).T

    elif (task == 2):

        if iterations > 90:
            raise ValueError('Maximum iteration count in Task 1 is 90')
            exit(1)

        # -------------------------------------------------------------
        # Classification task no.2 (Versicolor vs. Virginica)
        # -------------------------------------------------------------

        # -------------------- Training data --------------------------

        given_train_inputs = []

        for element in zip(data_train_versicolor, data_train_virginica):
            given_train_inputs.extend(element)

        given_train_inputs = np.array(given_train_inputs)

        given_train_outputs = np.zeros(90)

        # every second element is of class 1
        given_train_outputs[1::2] = 1
        given_train_outputs = given_train_outputs.reshape(1, -1).T

        # -------------------- Testing data --------------------------

        given_test_inputs = np.concatenate((
            data_test_versicolor, data_test_virginica))

        given_test_outputs = np.concatenate((
            np.zeros(5), np.ones(5))).reshape(1, -1).T
    else:
        raise ValueError('Incorrect task specified')

    # =================================================================
    # Running tests
    # =================================================================

    nn = NeuralNetwork()
    init_w = nn.synaptic_weights

    learning_rate = 0.8
    learning_rate_decimals = 1
    epochs = 1

    for i in range(2,135,2):
        iterations = i
        run(nn, init_w)
