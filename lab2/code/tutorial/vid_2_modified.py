#!/usr/bin/env python3

import numpy as np

class NeuralNetwork():

    def __init__(self):
        np.random.seed(1)
        # self.synaptic_weights = np.array([[0.0],[0.0],[0.0],[0.0],[0.0]])
        # self.synaptic_weights = np.array([[0.0],[0.0],[0.0]])
        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1

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
              training_inputs,
              training_outputs,
              learning_rate,
              activation_function,
              training_iterations,
              training_epochs):

        for epoch in range(training_epochs):
            for iteration in range(training_iterations):
                output = self.think(training_inputs, activation_function)
                # error = ((training_outputs - output)**2)/2
                error = training_outputs - output
                # adjustments = np.dot(training_inputs.T, error * (-1)*learning_rate)
                adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))
                # adjustments = np.dot(training_inputs.T, error * learning_rate)
                self.synaptic_weights = self.synaptic_weights + adjustments

    def think(self, inputs, activation_function):
        inputs = inputs.astype(float)
        if (activation_function == 'threshold'):
            output = self.threshold_fn(np.dot(inputs, self.synaptic_weights))
        elif (activation_function == 'sigmoid'):
            output = self.sigmoid_fn(np.dot(inputs, self.synaptic_weights))
        else:
            raise ValueError("Incorrect activationfunction specified")
            exit(1)

        return output

if __name__ == "__main__":
    neural_network = NeuralNetwork()
    # print("Random synaptic weights: ")
    print("Starting synaptic weights: ")
    print(neural_network.synaptic_weights)

    training_inputs = np.array([[0,0,1],
                                [1,1,1],
                                [1,0,1],
                                [0,1,1]])

    training_outputs = np.array([[0,1,1,0]]).T

    learning_rate = 1
    # activation_function = 'threshold'
    activation_function = 'sigmoid'
    iterations = 10000
    epochs = 1
    neural_network.train(training_inputs,
                         training_outputs,
                         learning_rate,
                         activation_function,
                         iterations,
                         epochs)

    print("Synaptic weights after training: ")
    print(neural_network.synaptic_weights)

    print("Outputs after training: ")
    print(neural_network.think(training_inputs, activation_function))

    testing_inputs = np.array([[0,0,1],
                               [1,0,0]])

    print("Test outputs: ")
    print(neural_network.think(testing_inputs, activation_function))

