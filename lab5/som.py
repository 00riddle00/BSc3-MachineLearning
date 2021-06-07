#!/usr/bin/env python
# Self organizing map (SOM)
# implementation

import math
import numpy as np


# =========================
# Helper functions
# =========================

# convert 1-indexed to 0-indexed
# ind = index
def ind(index):
    return index - 1


# calculate Euclidean distance
# between two numpy arrays
def euclidean_distance(a, b):
    print(a)
    print(type(a))
    print(b)
    print(type(b))
    exit()
    return np.linalg.norm(a - b)


# =========================
# Main functions
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


# check if (i,j) is in a vicinity to c
def N_c(c, i, j, t, e, n=3):
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


def eta_c(c, ij):
    d = euclidean_distance(c, ij)
    if d == 0:
        return 1

    return math.floor(d)


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


# =========================
# Main code
# =========================

# input vectors
X = np.array([[5.1, 3.5, 1.4, 0.2],
              [4.9, 3.0, 1.4, 0.2],
              [4.7, 3.2, 1.3, 0.2],
              [4.6, 3.1, 1.5, 0.0],
              [7.0, 3.2, 4.7, 1.4],
              [6.4, 3.2, 4.5, 1.5],
              [6.9, 3.1, 4.9, 1.5],
              [5.5, 2.3, 4.0, 1.0],
              [6.3, 3.3, 6.0, 2.5],
              [5.8, 2.7, 5.1, 1.9],
              [7.1, 3.0, 5.9, 2.1],
              [6.3, 2.9, 5.6, 1.8]], np.float16)

# classes of input vectors
labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], np.uint8)

e = 100  # epoch count
kx = 4  # grid rows
ky = 4  # grid columns
m = X.shape[0]  # no of input vectors
n = X.shape[1]  # attribute count of the input vector
M = np.random.rand(kx, ky, n)  # grid neurons

# collect euclidean distances
distances = np.zeros((kx, ky))

ROWS = kx
COLS = ky

# create neighbours grid
neighbours = {
    (p // COLS, p % COLS): [(p // COLS + x_inc - 1, p % COLS + y_inc - 1)
                            for x_inc in range(3) if 1 <= p // COLS + x_inc <= ROWS
                            for y_inc in range(3) if 1 <= p % COLS + y_inc <= COLS and not y_inc == x_inc == 1]
    for p in range(ROWS * COLS)}

# TODO def som_training(X_data, X_labels, M, e, kx, ky): -> return M
for t in range(1, e + 1):
    for l in range(1, m + 1):  # for every input vector 'l'
        for i in range(1, kx + 1):
            for j in range(1, ky + 1):
                # calculate euclidean distance
                distances[ind(i)][ind(j)] = \
                    euclidean_distance(M[ind(i)][ind(j)], X[ind(l)])
        # c = neuron leader
        c = np.unravel_index(distances.argmin(), distances.shape)
        # c = (c[0] + 1, c[1] + 1)  # restore 1-indexed notation
        # c = (c[0] + 1, c[1] + 1)  # restore 1-indexed notation
        for i in range(1, kx + 1):
            for j in range(1, ky + 1):
                print('i==', i, 'j==', j)
                # Update neurons using SOM learning rule
                Mij = M[ind(i)][ind(j)]
                M[ind(i)][ind(j)] = Mij + h(c, ind(i), ind(j), t, e) * (X[ind(l)] - Mij)

print(M)
