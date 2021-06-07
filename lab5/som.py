#!/usr/bin/env python

# ========================================
# Self organizing map (SOM) implementation
# (unsupervised learning)
# ========================================

import math
import numpy as np

global neighbours


# ================
# Helper functions
# ================

# convert 1-indexed to 0-indexed
# ind = index
def ind(index):
    return index - 1


# calculate Euclidean distance
# between two numpy arrays
def euclidean_distance(a, b):
    return np.linalg.norm(a - b)


# =================================================================
# Reading from a file
# =================================================================
def get_data():
    data = np.genfromtxt('dataset/iris.data', delimiter=',')

    data_setosa = data[:50]
    data_versicolor = data[50:100]
    data_virginica = data[100:150]

    data_train_setosa = data_setosa[:45]
    data_test_setosa = data_setosa[45:50]

    data_train_versicolor = data_versicolor[:45]
    data_test_versicolor = data_versicolor[45:50]

    data_train_virginica = data_virginica[:45]
    data_test_virginica = data_virginica[45:50]

    # -------------------- Training data --------------------------
    given_train_inputs = []

    for element in zip(*[data_train_setosa, data_train_versicolor,
                         data_train_virginica]):
        given_train_inputs.extend(element)

    given_train_inputs = np.array(given_train_inputs, np.float16)

    given_train_outputs = np.ones(135, np.uint8)

    # every second element is of class 1
    given_train_outputs[1::3] = 2
    given_train_outputs[2::3] = 3
    given_train_outputs = given_train_outputs.reshape(1, -1).T

    # -------------------- Testing data ---------------------------
    given_test_inputs = np.array(np.concatenate((
        data_test_setosa, data_test_versicolor, data_test_virginica, data)),
        np.float16)

    given_test_outputs = np.ones(15, np.uint8)
    given_test_outputs[5:10] = 2
    given_test_outputs[10:] = 3
    given_test_outputs = given_test_outputs.reshape(1, -1).T

    return [given_train_inputs,
            given_train_outputs,
            given_test_inputs,
            given_test_outputs]


# =========================
# Main program functions
# =========================

# learning rate function
# @param t - current iteration number
# @param T - total number of iterations (epochs)
def alpha_fn(t, T, kind=1):
    if kind not in [1, 2, 3]:
        raise ValueError("kind must be one of 1, 2, 3")
    if kind == 1:
        return 1 / t
    elif kind == 2:
        return 1 - (t / T)
    else:
        return math.pow(0.005, (t / T))


def eta_c(c, ij):
    d = euclidean_distance(c, ij)
    if d == 0:
        return 1

    return math.floor(d)


# check if (i,j) is in a vicinity to c
def N_c(c, i, j, t, e, n=3):
    global neighbours

    vicinity = 0

    vicinity_neighbours = set(neighbours[c])
    vicinity_neighbours.add(c)

    if (i, j) in vicinity_neighbours:
        return True

    for _i in range(1, n + 1):
        if t / e < _i / n:
            vicinity = n - _i + 1
            break

    for _i in range(2, vicinity + 1):
        for v_prev_neighbour in vicinity_neighbours.copy():
            v_neighbours = neighbours[v_prev_neighbour]
            for v_neighbour in v_neighbours:
                vicinity_neighbours.add(v_neighbour)
                if (i, j) == v_neighbour:
                    return True

    return False


# neighbour function
# @param c - neuron leader
# @params i,j - indices (1-indexed)
# @param t - current iteration number
# @param e - total number of iterations (epochs)
def h(c, i, j, t, e, kind='bubble'):
    if kind not in ['bubble', 'gaussian']:
        raise ValueError("kind must be one of 'bubble', 'gaussian'")
    if kind == 'bubble':
        if N_c(c, i, j, t, e):
            return alpha_fn(t, e)
        else:
            return 0
    elif kind == 'gaussian':
        Rc = np.asarray(c)
        Rij = np.asarray((i, j))
        nominator = - math.pow(euclidean_distance(Rc, Rij), 2)
        denominator = 2 * math.pow(eta_c(Rc, Rij), 2)
        return alpha_fn(t, e) * np.exp(nominator / denominator)


def som_train(X, M, e, kx, ky):
    distances = np.zeros((kx, ky))
    for t in range(1, e + 1):
        print(f'Epoch {t}')
        for l in range(1, m + 1):  # for every input vector 'l'
            for i in range(1, kx + 1):
                for j in range(1, ky + 1):
                    # calculate euclidean distance
                    distances[ind(i)][ind(j)] = \
                        euclidean_distance(M[ind(i)][ind(j)], X[ind(l)])
            # c = neuron leader
            c = np.unravel_index(distances.argmin(), distances.shape)
            for i in range(1, kx + 1):
                for j in range(1, ky + 1):
                    # Update neurons using SOM learning rule
                    Mij = M[ind(i)][ind(j)]
                    M[ind(i)][ind(j)] = Mij + h(c, ind(i), ind(j), t, e) * (X[ind(l)] - Mij)
    return M


# =========================
# Main code
# =========================
if __name__ == '__main__':
    data = get_data()

    # train inputs (vectors)
    X = data[0]
    X = np.delete(X, 4, axis=1)

    test_inputs = data[2]
    test_labels = data[3]

    m = X.shape[0]  # no of input vectors
    n = X.shape[1]  # how many attributes the input vector has

    kx = 8  # grid rows
    ky = 8  # grid columns
    np.random.seed(0)
    M = np.random.rand(kx, ky, n)  # grid neurons

    epochs = 20

    # create neighbours grid
    neighbours = {
        (p // ky, p % ky): [(p // ky + x_inc - 1, p % ky + y_inc - 1)
                            for x_inc in range(3) if 1 <= p // ky + x_inc <= kx
                            for y_inc in range(3) if 1 <= p % ky + y_inc <= ky and not y_inc == x_inc == 1]
        for p in range(kx * ky)}

    trained_M = som_train(X, M, epochs, kx, ky)
    print(trained_M)
