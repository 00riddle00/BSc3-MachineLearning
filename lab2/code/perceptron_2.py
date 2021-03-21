#!/usr/bin/env python3

import numpy as np

def debug(text):
    global debug
    if DEBUG_FLAG:
        print(text)

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

                # adjustments = np.dot(
                    # train_inputs.T, error * self.sigmoid_derivative(output))

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
    DEBUG_FLAG = True

    # activation function is one of {'threshold', 'sigmoid'}
    activation_function = 'threshold'
    learning_rate = 1
    iterations = 10000
    epochs = 1
    # =================================================================

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
        # -------------------------------------------------------------
        # Classification task no.1 (Setosa vs. Versicolor_&_Virginica)
        # -------------------------------------------------------------
        given_train_inputs = np.concatenate((
            data_train_setosa, data_train_versicolor_and_virginica))

        given_train_outputs = np.concatenate((
            np.zeros(45), np.ones(90))).reshape(1, -1).T

        given_test_inputs = np.concatenate((
            data_test_setosa, data_test_versicolor_and_virginica))

        given_test_outputs = np.concatenate((
            np.zeros(5), np.ones(10))).reshape(1, -1).T
    elif (task == 2):
        # -------------------------------------------------------------
        # Classification task no.2 (Versicolor vs. Virginica)
        # -------------------------------------------------------------
        given_train_inputs = np.concatenate((
            data_train_versicolor, data_train_virginica))

        given_train_outputs = np.concatenate((
            np.zeros(45), np.ones(45))).reshape(1, -1).T

        given_test_inputs = np.concatenate((
            data_test_versicolor, data_test_virginica))

        given_test_outputs = np.concatenate((
            np.zeros(5), np.ones(5))).reshape(1, -1).T
    else:
        raise ValueError('Incorrect task specified')

    # =================================================================

    neural_network = NeuralNetwork()

    debug('Random starting synaptic weights: ')
    debug(np.round(neural_network.synaptic_weights, 2))

    neural_network.train(given_train_inputs,
                         given_train_outputs,
                         learning_rate,
                         activation_function,
                         iterations,
                         epochs)

    train_outputs = neural_network.think(given_train_inputs, activation_function)
    test_outputs = neural_network.think(given_test_inputs, activation_function)

    debug(f'Outputs after training:\n {train_outputs}')
    debug(f'Test outputs:\n {test_outputs}')

    # let NaN mean that the item was not assigned to any class
    if activation_function == 'sigmoid':
        train_outputs = np.array([
            1 if xi > 0.9 else 0 if xi < 0.1 else np.nan for xi in train_outputs])

        test_outputs = np.array([
            1 if xi > 0.9 else 0 if xi < 0.1 else np.nan for xi in test_outputs])

    train_results = np.array([sum(x) for x in zip(
        given_train_outputs.flatten(), train_outputs)])

    test_results = np.array([sum(x) for x in zip(
        given_test_outputs.flatten(), test_outputs)])

    debug(f'Training results:\n {train_results}')
    debug(f'Test results:\n {test_results}')

    # if the sum of the given class and resulting class is 1, that means
    # misclassification has occured.
    # if, however, the sum is nan, the item was not assigned to any class
    # else, the sum values of 0 and 2 means correct classification.
    unique, counts = np.unique(train_results[~np.isnan(train_results)], return_counts=True)
    train_results = dict(zip(unique, counts))

    unique, counts = np.unique(test_results[~np.isnan(test_results)], return_counts=True)
    test_results = dict(zip(unique, counts))

    debug(f'Training results (summary):\n {train_results}')
    debug(f'Test results (summary):\n {test_results}')

    train_accuracy = (train_results[0.0] + train_results[2.0]) / len(train_outputs)
    test_accuracy = (test_results[0.0] + test_results[2.0]) / len(test_outputs)

    print('==============================')

    print('Synaptic weights after training: ')
    print(np.round(neural_network.synaptic_weights, 2))

    print('==============================')
    print('Parameters')
    print('==============================')
    print('Activation function:', activation_function)
    print('Learning rate:', learning_rate)
    print('Epochs:', epochs)
    print('Iterations per epoch:', iterations)
    print('==============================')

    print('Training accuracy: {:.2f}'.format(train_accuracy))
    print('Testing accuracy: {:.2f}'.format(test_accuracy))
    print('==============================')

