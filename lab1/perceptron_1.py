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

def sigmoid_fn(x):
    sig = 1 / (1 + np.exp(-x))     # Define sigmoid function
    sig = np.minimum(sig, 0.9999)  # Set upper bound
    sig = np.maximum(sig, 0.0001)  # Set lower bound
    return sig

training_inputs = np.array([[-0.2, 0.5,-1],
                            [0.2, -0.5,-1],
                            [0.8, -0.8,-1],
                            [0.8, 0.8,-1]])

training_outputs = [0, 0, 1, 1]

synaptic_weights = np.array([[1.0],[1.0],[1.0]])

# USAGE: change one to True, and another to False
threshold = True
sigmoid = False

for i in range(1,1000001):
    outputs = np.dot(training_inputs, synaptic_weights)

    if threshold:
        outputs = threshold_fn(outputs.tolist())

        if outputs == training_outputs:
            break
    elif sigmoid:
        outputs = sigmoid_fn(outputs)
        out = outputs.flatten().tolist()

        for j in range(2):
            if out[j] < 0.25:
                out[j] = 0

        for j in range(2,4):
            if out[j] > 0.75:
                out[j] = 1

        if out == training_outputs:
            break
    else:
        print('ERROR: unspecified activation function')
        exit(1)

    # update weights
    if i % 10000 == 0:
        synaptic_weights[0][0] += 0.2
        synaptic_weights[1][0] = 1.0
        synaptic_weights[2][0] = 1.0
    elif i % 100 == 0:
        synaptic_weights[1][0] += 0.2
        synaptic_weights[2][0] = 1.0
    else:
        synaptic_weights[2][0] += 0.2

print('Suitable synaptic weights:')
print(synaptic_weights)

print('Corresponding outputs:')
print(outputs)
