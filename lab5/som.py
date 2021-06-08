#!/usr/bin/env python

# ========================================
# Self organizing map (SOM) implementation
# (unsupervised learning)
# ========================================

import math
import numpy as np

global VICINITIES, H_FN_KIND, ALPHA_FN_KIND
global RANDOM_SEED
global SHOW_CLASS_NAMES
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
def get_data(test_data_percentage=10, all=False):
    if not all in [True, False]:
        raise ValueError("'all' parameter should be either True or False")

    data = np.genfromtxt('dataset/iris.data', delimiter=',')

    if all:
        data_train_setosa = data[:50]
        data_train_versicolor = data[50:100]
        data_train_virginica = data[100:150]

        # -------------------- Training data --------------------------
        train_inputs = []

        for element in zip(*[data_train_setosa, data_train_versicolor,
                             data_train_virginica]):
            train_inputs.extend(element)

        train_inputs = np.array(train_inputs, np.float16)

        train_outputs = np.ones(150, np.uint8)

        # classes 1, 2, 3, repeat for every three elements
        train_outputs[1::3] = 2
        train_outputs[2::3] = 3
        train_outputs = train_outputs.reshape(1, -1).T

        test_inputs = train_inputs.copy()
        test_outputs = train_outputs.copy()

        return [train_inputs,
                train_outputs,
                test_inputs,
                test_outputs]

    # here start the case when all = False
    if not isinstance(test_data_percentage, int):
        raise ValueError("test_data_percentage parameter should be an integer")
    elif test_data_percentage not in range(1, 51):
        raise ValueError("test_data_percentage values "
                         "can only be between 1 and 50")

    data = np.genfromtxt('dataset/iris.data', delimiter=',')

    data_setosa = data[:50]
    data_versicolor = data[50:100]
    data_virginica = data[100:150]

    # ---------------------------------------------------------------
    train_data_percentage = 100 - test_data_percentage
    train_data_count = int(150 * train_data_percentage / 100)

    train_data_div_3 = int(train_data_count / 3)

    train_data_each = [train_data_div_3,
                       train_data_div_3,
                       train_data_count - (2 * train_data_div_3)]

    tmp = train_data_each

    data_train_setosa = data_setosa[:tmp[0]]
    data_test_setosa = data_setosa[tmp[0]:]

    data_train_versicolor = data_versicolor[:tmp[1]]
    data_test_versicolor = data_versicolor[tmp[1]:]

    data_train_virginica = data_virginica[:tmp[2]]
    data_test_virginica = data_virginica[tmp[2]:]

    test_data_each = [data_test_setosa.shape[0],
                      data_test_versicolor.shape[0],
                      data_test_virginica.shape[0]]

    test_data_count = sum(test_data_each)

    # -------------------- Training data --------------------------
    train_inputs = []

    for element in zip(*[data_train_setosa, data_train_versicolor,
                         data_train_virginica]):
        train_inputs.extend(element)

    train_inputs = np.array(train_inputs, np.float16)

    train_outputs = np.ones(train_data_count, np.uint8)

    # classes 1, 2, 3, repeat for every three elements
    train_outputs[1::3] = 2
    train_outputs[2::3] = 3
    train_outputs = train_outputs.reshape(1, -1).T

    # -------------------- Testing data ---------------------------
    test_inputs = np.array(np.concatenate((
        data_test_setosa, data_test_versicolor, data_test_virginica)),
        np.float16)

    test_outputs = np.ones(test_data_count, np.uint8)
    tmp = test_data_each

    test_outputs[tmp[0]:tmp[0] + tmp[1]] = 2
    test_outputs[tmp[0] + tmp[1]:] = 3
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

    # TODO can be simplified using euclidean
    # distance (as in 'eta_c' function)
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
# @param kind - one of 'bubble', 'gaussian'
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
        # (neuron leader tuple): [[input's class, input's number], [..., ...]]
        results.setdefault(c, []).append([Y_labels[l - 1][0], l])

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


def som_draw(results, kx, ky, show_class_names=True):
    # in results, class name is at index 0,
    # vector number is at index 1
    index = int(show_class_names is False)
    separator = index * ','

    grid = [['' for y in range(ky)] for x in range(kx)]

    max_len = 0

    for x in range(kx):
        for y in range(ky):
            if (x, y) in results:
                grid[x][y] = f'{separator}'.join('{}'.format(
                    result[index]) for result in results[(x, y)])

                if len(grid[x][y]) > max_len:
                    max_len = len(grid[x][y])

    # required for tidy column number printing
    if max_len % 2 == 0:
        print(
            f"   {''.join(['{1}{0}{1}'.format(col_number, ' ' * (((max_len + 2) // 2) - int(col_number > 10))) for col_number in range(1, ky + 1)])}")
    else:
        print(
            f"   {''.join(['{1}{0}{2}'.format(col_number, ' ' * (((max_len + 2) // 2 + 1) - int(col_number > 10)), ' ' * (((max_len + 2) // 2) - int(col_number > 10))) for col_number in range(1, ky + 1)])}")

    # print grid's upper bar
    print(f"   {'_' * (((max_len + 3) * ky) + 1)}")

    # print grid's rows
    for i, row in enumerate(grid, 1):
        print(f"{str(i).rjust(2)} |{''.join(['{0:_>{1}}_|'.format(grid_value, max_len + 1) for grid_value in row])}")


# =========================
# Main code
# =========================
if __name__ == '__main__':
    # choose one function out of two:
    #
    # get_data()     - get separate training and testing data
    #                  (see function itself for train/test distribution).
    # get_data_all() - use all data for training and the same data for
    #                  testing as well.
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
    # Grid size (rows and columns).
    # Can be a rectangle as well.
    kx = 10
    ky = 10

    epochs = 10

    # Vicinities parameter denotes how far away does a
    # neighbour "count" as a neighbour at the beginning of
    # the SOM learning. This number will be decremented
    # gradually as the learning progresses and will eventually
    # reach a value of 1 (only for closest neighbours)
    VICINITIES = 3
    # One of 'bubble', 'gaussian'
    H_FN_KIND = 'bubble'
    # One of 'simple_div', 'simple_div_sub', 'power'
    ALPHA_FN_KIND = 'simple_div'

    # choose for different random
    # filling of the initial grid
    RANDOM_SEED = 0

    # Visualization parameters

    # Set it to true to show input vectors' class names
    # in the resulting grid. Otherwise the number of
    # the input vector will be shown.
    SHOW_CLASS_NAMES = True

    # Set it to true to print out SOM neurons' weights
    # and winner neurons for every input vector
    VERBOSE_OUTPUT = False

    # ------------------------------------
    # Set up
    # ------------------------------------
    n = X.shape[1]  # how many attributes the input vector has
    np.random.seed(RANDOM_SEED)
    M = np.random.rand(kx, ky, n)  # grid neurons

    # for every cell in the grid, indicate its neighbours
    NEIGHBOURS = {
        (p // ky, p % ky): [(p // ky + x_inc - 1, p % ky + y_inc - 1)
                            for x_inc in range(3) if 1 <= p // ky + x_inc <= kx
                            for y_inc in range(3) if 1 <= p % ky + y_inc <= ky and not y_inc == x_inc == 1]
        for p in range(kx * ky)}

    # ------------------------------------
    # SOM in action
    # ------------------------------------
    M_trained = som_train(X, M, epochs, kx, ky)

    results = som_test(Y, Y_labels, M_trained, kx, ky)

    print('-' * 40)
    print('Parameters:')
    print('-' * 40)
    print('Epochs: ', epochs)
    print(f'Grid size: {kx}x{ky}')
    print('h (neighbourhood function) type: ', H_FN_KIND)
    print('alpha function type: ', ALPHA_FN_KIND)
    print('Starting vicinities: ', VICINITIES)
    print('Random seed: ', RANDOM_SEED)
    print('-' * 40)

    q_error, t_error = som_evaluate(X, M_trained, kx, ky)
    print('Quantization error: ', q_error)
    print('Topographic error: ', t_error)
    print('-' * 40)

    print('\nResulting grid:')
    if SHOW_CLASS_NAMES:
        print("(showing input vectors' classes)\n")
    else:
        print("(showing input vectors' numbers)\n")

    som_draw(results, kx, ky, show_class_names=SHOW_CLASS_NAMES)

    if VERBOSE_OUTPUT:
        print()
        print('.' * 40)
        print('Training statistics:')
        print('.' * 40)
        print()
        print('Grid after training:\n')
        print(M)
        print()

        print("Winner neurons together with a list of pairs indicating\n"
              "associated input vectors' classes and their numbers:\n")
        for winner_neuron in results:
            print(f'{winner_neuron}: {results[winner_neuron]}')
