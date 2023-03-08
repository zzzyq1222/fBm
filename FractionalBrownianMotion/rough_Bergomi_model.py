"""
Simulate the rough Bergomi model based on
Markovian approximation of the rough Bergomi model for Monte Carlo option pricing 5.1 & 5.2
"""
import math

import numpy as np
import sys

sys.path.append('..')
import utils
import hosking_method as hm

"""
generate Volterra process X
"""
def volterra_process(timespan, interval):
    alpha = -0.43
    eta = 1.9
    size = int(timespan / interval)
    # todo H = alpha+0.5
    H = 0.5 + alpha

    fBm = hm.hoskingMethodFBm()
    # simulate W_tj
    Wt = fBm.generateFBm(timespan, interval, 0.5)

    # simulate W_t_1
    Wt_1 = [((interval * 0.5) ** alpha) * Wt[0]]
    for i in range(1, size):
        newW = ((interval * 0.5) ** alpha) * (Wt[i] - Wt[i - 1])
        Wt_1.append(newW)

    # simulate W_j
    W_j = [Wt[0]]
    for i in range(1, size):
        W_j.append(Wt[i] - Wt[i - 1])

    # calculate X

    sigma = math.sqrt(2 * alpha + 1)
    b_star = lambda k: ((k ** (alpha + 1) - (k - 1) ** (alpha + 1)) / (alpha + 1)) ** (1 / alpha)
    g = lambda b, H: (b * interval) ** (H - 0.5)

    # column of sigma
    sigma_col = np.ones(size) * sigma
    # construct the matrix W
    W = np.zeros((size, size))
    W[0, :] = Wt_1[::-1]
    W_j.reverse()
    W_j = np.asarray(W_j)
    for i in range(1, size):
        # todo H
        W_j = np.delete(W_j, 0)
        W[i, :(size - i)] = W_j * g(b_star(i + 1), H)

    # rotate anti-clockwise to get the W
    W = np.rot90(W, 1)
    X = np.dot(W, sigma_col)

    return X, Wt


"""
simulation of the stock price in the rBergomi model
"""


def simulate_rBergomi_model(X, Wt, timespan, interval):
    N = int(timespan / interval)
    Xi_0 = 0.026
    eta = 1.9
    alpha = -0.43

    # generate V
    V = [Xi_0]
    for i in range(1, N):
        exponent = eta * X[i] - (eta ** 2) * 0.5 * (i * interval) ** (2 * alpha + 1)
        Vt = Xi_0 * (math.e ** exponent)
        V.append(Vt)

    # generate log stock price
    logS = [0]
    for i in range(1, N):
        newS = logS[i - 1] + math.sqrt(V[i]) * (Wt[i] - Wt[i - 1]) - 0.5 * V[i] * interval
        logS.append(newS)

    S = [math.e ** s for s in logS]

    return logS, S


if __name__ == '__main__':
    timespan = 10
    interval = 0.01
    res = []
    for i in range(5):
        X, Wt = volterra_process(timespan, interval)
        logS, S = simulate_rBergomi_model(X, Wt, timespan, interval)
        res.append(S)
    utils.draw_n_paths(5, timespan, interval, res)
