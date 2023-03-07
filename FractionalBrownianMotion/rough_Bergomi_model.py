"""
Simulate the rough Bergomi model based on
Markovian approximation of the rough Bergomi model for Monte Carlo option pricing 5.1 & 5.2
"""
import math

import numpy as np
import sys

sys.path.append('..')
import utils
import cholesky_method as cm

"""
generate Volterra process X
"""


def volterra_process(timespan, interval):
    alpha = -0.43
    eta = 1.9
    size = int(timespan / interval)
    #todo H = alpha+0.5
    H = 0.5 + alpha

    fBm = cm.choleskyMethodfBM()
    # simulate W_tj
    # Wt = fBm.generateFBm(size, interval, 0.5)
    Wt = fBm.generateFBm(size, interval, 0.5)
    # simulate W_t_1
    Wt_1 = []
    for i in range(size):
        newW = ((interval * 0.5) ** alpha) * (Wt[i] - Wt[i-1])
        Wt_1.append(newW)

    # simulate W_j
    # 反正是N-1个，所以直接从1开始
    W_j = []
    for i in range(1, size):
        W_j.append(Wt[i] - Wt[i - 1])

    # calculate X
    #todo sigma
    sigma = math.sqrt(2 * alpha + 1)
    b_star = lambda k: ((k ** (alpha + 1) - (k - 1) ** (alpha + 1)) / (alpha + 1)) ** (1 / alpha)
    g = lambda b, H: (b * interval) ** (H - 0.5)
    # column of sigma
    sigma_col = np.ones(size) * sigma
    W = np.zeros((size, size))
    W[0, :] = Wt_1[::-1]
    W_j.reverse()
    W_j = np.asarray(W_j)
    for i in range(1, size):
        # todo H
        W[i, :(size-i)] = W_j[:(size - i)] * g(b_star(i + 1), H)

    # rotate anti-clockwise
    W = np.rot90(W, 1)
    X = W.dot(sigma_col)

    return X, Wt

"""
input: Volterra process X
"""
def simulate_rBergomi_model(X, Wt,timespan, interval):
    N = int(timespan/interval)
    Xi_0 = 0.026
    eta = 1.9
    alpha = -0.43

    V = []
    for i in range(N):
        exponent = eta * X[i] - (eta**2) * 0.5 * ((i+1)*interval)**(2*alpha+1)
        Vt = Xi_0 * (math.e ** exponent)
        V.append(Vt)

    logS = [0]
    for i in range(1,N):
        newS = logS[i-1] + math.sqrt(V[i]) * (Wt[i] - Wt[i-1]) - 0.5 * V[i]
        logS.append(newS)

    S = [math.e ** s for s in logS]
    return logS, S


if __name__ == '__main__':
    timespan = 10
    interval = 0.01
    X, Wt = volterra_process(timespan, interval)
    logS, S = simulate_rBergomi_model(X, Wt, timespan, interval)
    utils.draw_n_paths(1, timespan, interval, [S])

