#!/usr/bin/env python

# ========================================
# Self organizing map (SOM) implementation
# (unsupervised learning)
# ========================================

import math
import numpy as np

global VICINITIES, H_FN_KIND, ALPHA_FN_KIND
global NEIGHBOURS


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
    train_inputs = []

    for element in zip(*[data_train_setosa, data_train_versicolor,
                         data_train_virginica]):
        train_inputs.extend(element)

    train_inputs = np.array(train_inputs, np.float16)

    train_outputs = np.ones(135, np.uint8)

    # every second element is of class 1
    train_outputs[1::3] = 2
    train_outputs[2::3] = 3
    train_outputs = train_outputs.reshape(1, -1).T

    # -------------------- Testing data ---------------------------
    test_inputs = np.array(np.concatenate((
        data_test_setosa, data_test_versicolor, data_test_virginica)),
        np.float16)

    test_outputs = np.ones(15, np.uint8)
    test_outputs[5:10] = 2
    test_outputs[10:] = 3
    test_outputs = test_outputs.reshape(1, -1).T

    return [train_inputs,
            train_outputs,
            test_inputs,
            test_outputs]


# =========================
# Main program functions
# =========================

# learning rate function
# @param t - current iteration number
# @param T - total number of iterations (epochs)
def alpha_fn(t, T, kind='simple_div'):
    kinds = ['simple_div', 'simple_div_sub', 'power']
    if kind == kinds[0]:
        return 1 / t
    elif kind == kinds[1]:
        return 1 - (t / T)
    elif kind == kinds[2]:
        return math.pow(0.005, (t / T))
    else:
        if kind not in kinds:
            raise ValueError(f"alpha_fn must be one of "
                             f"{', '.join([kind for kind in kinds])}")


def eta_c(c, ij):
    d = euclidean_distance(c, ij)
    if d == 0:
        return 1

    return math.floor(d)


# check if (i,j) is in a vicinity to c
def N_c(c, i, j, t, e, n=3):
    global NEIGHBOURS

    vicinity = 0

    vicinity_neighbours = set(NEIGHBOURS[c])
    vicinity_neighbours.add(c)

    if (i, j) in vicinity_neighbours:
        return True

    for _i in range(1, n + 1):
        if t / e < _i / n:
            vicinity = n - _i + 1
            break

    # TODO can be simplified with euclidean distance (see eta_c function)
    for _i in range(2, vicinity + 1):
        for v_prev_neighbour in vicinity_neighbours.copy():
            v_neighbours = NEIGHBOURS[v_prev_neighbour]
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
def h_fn(c, i, j, t, e, kind='bubble'):
    global VICINITIES, ALPHA_FN_KIND
    kinds = ['bubble', 'gaussian']
    if kind == kinds[0]:
        if N_c(c, i, j, t, e, n=VICINITIES):
            return alpha_fn(t, e, kind=ALPHA_FN_KIND)
        else:
            return 0
    elif kind == kinds[1]:
        Rc = np.asarray(c)
        Rij = np.asarray((i, j))
        nom = - math.pow(euclidean_distance(Rc, Rij), 2)
        denom = 2 * math.pow(eta_c(Rc, Rij), 2)
        return alpha_fn(t, e, kind=ALPHA_FN_KIND) * np.exp(nom / denom)
    else:
        raise ValueError(f"h_fn must be one of "
                         f"{', '.join([kind for kind in kinds])}")


def som_train(X, M, e, kx, ky):
    global H_FN_KIND
    m = X.shape[0]  # no of input vectors
    print(f'Learning...')
    for t in range(1, e + 1):
        print(f'Epoch {t}')
        for l in range(1, m + 1):  # for every input vector 'l'
            distances = np.zeros((kx, ky))
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
                    M[ind(i)][ind(j)] = \
                        Mij + \
                        h_fn(c, ind(i), ind(j), t, e, kind=H_FN_KIND) \
                        * (X[ind(l)] - Mij)
    print('Done.\n')
    return M


def som_test(Y, Y_labels, M_t, kx, ky):
    m = Y.shape[0]  # no of input vectors
    results = {}
    for l in range(1, m + 1):  # for every input vector 'l'
        distances = np.zeros((kx, ky))
        for i in range(1, kx + 1):
            for j in range(1, ky + 1):
                # calculate euclidean distance
                distances[ind(i)][ind(j)] = \
                    euclidean_distance(M_t[ind(i)][ind(j)], Y[ind(l)])
        # c = neuron leader
        c = np.unravel_index(distances.argmin(), distances.shape)
        # (neuron leader tuple): [[input's number, input's class], [..., ...]]
        results.setdefault(c, []).append([l, Y_labels[l - 1][0]])

    return results


def som_evaluate(X, M_t, kx, ky):
    q_error = 0
    t_error = 0
    m = X.shape[0]
    for l in range(1, m + 1):
        distances = np.zeros((kx, ky))
        for i in range(1, kx + 1):
            for j in range(1, ky + 1):
                distances[ind(i)][ind(j)] = \
                    euclidean_distance(M_t[ind(i)][ind(j)], X[ind(l)])
        c = np.unravel_index(distances.argmin(), distances.shape)

        q_error += distances[c[0]][c[1]]

        distances[c[0]][c[1]] = math.inf
        second_best_c = np.unravel_index(distances.argmin(), distances.shape)
        if euclidean_distance(np.asarray(c), np.asarray(second_best_c)) >= 2:
            t_error += 1

    return q_error / m, t_error / m


def som_draw(results, kx, ky):
    R = [['' for y in range(ky)] for x in range(kx)]

    max_len = 0
    for x in range(kx):
        for y in range(ky):
            if (x, y) in results:
                R[x][y] = ''.join('{}'.format(
                    item[1]) for item in results[(x, y)])

                if len(R[x][y]) > max_len:
                    max_len = len(R[x][y])

    if max_len % 2 == 0:
        print(f"  {''.join(['{1}{0}{1}'.format(item, ' ' * ((max_len + 2) // 2)) for item in range(1, ky + 1)])}")
    else:
        print(
            f"  {''.join(['{1}{0}{2}'.format(item, ' ' * ((max_len + 2) // 2 + 1), ' ' * ((max_len + 2) // 2)) for item in range(1, ky + 1)])}")

    print(f"  {'_' * (((max_len + 3) * ky) + 1)}")

    for i, row in enumerate(R, 1):
        print(f"{i} |{''.join(['{0:_>{1}}_|'.format(item, max_len + 1) for item in row])}")


# =========================
# Main code
# =========================
if __name__ == '__main__':
    data = get_data()

    # ------------------------------------
    # Set inputs
    # ------------------------------------
    # train inputs
    X = data[0]
    X = np.delete(X, 4, axis=1)

    # test inputs and labels
    Y = data[2]
    Y = np.delete(Y, 4, axis=1)
    Y_labels = data[3]

    # ------------------------------------
    # Changeable parameters
    # ------------------------------------
    kx = 8  # grid rows
    ky = 8  # grid columns
    epochs = 10
    VICINITIES = 3
    H_FN_KIND = 'bubble'
    ALPHA_FN_KIND = 'simple_div'

    # ------------------------------------
    # Set up
    # ------------------------------------
    n = X.shape[1]  # how many attributes the input vector has
    np.random.seed(0)
    M = np.random.rand(kx, ky, n)  # grid neurons

    # create neighbours grid
    NEIGHBOURS = {
        (p // ky, p % ky): [(p // ky + x_inc - 1, p % ky + y_inc - 1)
                            for x_inc in range(3) if 1 <= p // ky + x_inc <= kx
                            for y_inc in range(3) if 1 <= p % ky + y_inc <= ky and not y_inc == x_inc == 1]
        for p in range(kx * ky)}

    # ------------------------------------
    # SOM in action
    # ------------------------------------
    M_trained = som_train(X, M, epochs, kx, ky)

    q_error, t_error = som_evaluate(X, M_trained, kx, ky)
    print('Quantization error: ', q_error)
    print('Topographic error: ', t_error)

    results = som_test(Y, Y_labels, M_trained, kx, ky)

    print('\nResulting grid:\n')
    som_draw(results, kx, ky)
