#!/usr/bin/env python3

import numpy as np

def threshold_fn(x):
    output = []
    for el in x:
        if el[0] > 0:
            output.append(1)
        else:
            output.append(0)
    return output

def sigmoid(x):
    sig = 1 / (1 + np.exp(-x))     # Define sigmoid function
    sig = np.minimum(sig, 0.9999)  # Set upper bound
    sig = np.maximum(sig, 0.0001)  # Set lower bound
    return sig

def sigmoid_derivative(x):
    return x * (1 - x)

training_inputs = np.array([[0,0,1],
                            [1,1,1],
                            [1,0,1],
                            [0,1,1]])

# training_outputs = [0, 0, 1, 1]
training_outputs = np.array([[0,1,1,0]]).T

np.random.seed(1)

# synaptic_weights = np.array([[1.0],[1.0],[1.0]])
synaptic_weights = 2 * np.random.random((3, 1)) - 1

print('Random starting synaptic weights: ')
print(synaptic_weights)

# USAGE: change one to True, and another to False
# threshold = False
# sigmoid = True

for iteration in range(1000):

    # if threshold:
        # outputs = threshold_fn(outputs.tolist())
        # if outputs == training_outputs:
            # break
    # elif sigmoid:
        # outputs = sigmoid_fn(outputs)
        # out = outputs.flatten().tolist()

        # for j in range(2):
            # if out[j] < 0.25:
                # out[j] = 0

        # for j in range(2,4):
            # if out[j] > 0.75:
                # out[j] = 1

        # if out == training_outputs:
            # break
    # else:
        # print("ERROR: unspecified activation function")
        # exit(1)

    input_layer = training_inputs

    outputs = sigmoid(np.dot(training_inputs, synaptic_weights))

    error = training_outputs - outputs

    adjustments = error * sigmoid_derivative(outputs)

    synaptic_weights += np.dot(input_layer.T, adjustments)

print('Synaptic weights after training:')
print(synaptic_weights)

print('Outputs after training: ')
print(outputs)
