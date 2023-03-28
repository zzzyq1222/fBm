import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import brentq

import sys

sys.path.append('..')
import utils
import GaussianNoise.gaussian_noise_simulation as gn

"""
Simulate the rough Bergomi model based on
Markovian approximation of the rough Bergomi model for Monte Carlo option pricing 5.1 & 5.2
eta = 1.9
alpha = -0.43
xi_0 = 0.026

rho = -0.9
xi_0 = 0.235 ** 2
parameter rho is from paper: Hybrid scheme for Brownian semistationary processes
"""


# generate Volterra Process
# N: number of paths
def VolterraProcess(N, steps, interval):
    alpha = -0.43

    # todo
    cov = lambda a: np.array([[interval, 1. / (a + 1) * interval ** (a + 1)],
                              [1. / (a + 1) * interval ** (a + 1),
                               1. / (2 * a + 1) * interval ** (2 * a + 1)]])
    b = lambda k: ((k ** (alpha + 1) - (k - 1) ** (alpha + 1)) / (alpha + 1)) ** (1 / alpha)
    g = lambda x: x ** alpha
    sigma = np.sqrt(2 * alpha + 1)

    # mean, cov ,size
    dW = np.random.multivariate_normal(np.array([0, 0]), cov(alpha), (N, steps))

    # Riemann sum
    I1 = np.zeros((N, 1 + steps))
    for i in range(0, steps):
        I1[:, i + 1] = dW[:, i, 1]

    # Wiener integral
    kernel = np.zeros(1 + steps)
    for i in range(2, steps):
        kernel[i] = g(b(i) / steps)

    W = dW[:, :, 0]
    tmp = np.zeros((N, len(W[0, :]) + len(kernel) - 1))
    for i in range(N):
        tmp[i, :] = np.convolve(kernel, W[i, :])
    I2 = tmp[:, :steps]
    VP = sigma * (I1[:, 0:steps] + I2)
    return VP, dW


def simulate_rBergomi_model(VP, dW, timespan, interval, steps):
    Xi_0 = 0.026
    eta = 1.9
    alpha = -0.43

    # construct variance process
    t = np.array([i * interval for i in range(steps)])
    dW2 = np.random.randn(N, steps) * np.sqrt(interval)

    rho = -0.9
    dB = rho * dW[:, :, 0] + np.sqrt(1 - rho ** 2) * dW2
    spotV = Xi_0 * np.exp(eta * VP - 0.5 * (eta ** 2) * t ** (2 * alpha + 1))

    # generate logS and S
    dlogS = np.sqrt(spotV) * dB - 0.5 * spotV * interval
    S = np.exp(np.cumsum(dlogS, axis=1))
    S = np.insert(S, 0, 1, axis=1)

    return spotV, S, t


def VolterraProcessTest(V, timespan, interval, steps):
    alpha = -0.43
    # known expectation and variance
    t = np.array([i * interval for i in range(steps)])
    E = 0 * t
    Var = t ** (2 * alpha + 1)

    # simulated data
    sim_E = np.mean(V, axis=0, keepdims=True)
    sim_Var = np.var(V, axis=0, keepdims=True)
    plot, ax = plt.subplots()
    ax.plot(t, E)
    ax.plot(t, Var)
    ax.plot(t, sim_E[0, :])
    ax.plot(t, sim_Var[0, :])

    ax.set_ylabel('X')
    ax.set_xlabel('t')
    plt.title('Mean and variance of Volterra process')
    plt.show()


def VtTest(Vt, timespan, interval):
    size = int(timespan / interval)
    t = np.array([i * interval for i in range(size)])
    Vt_mean = np.mean(Vt, axis=0, keepdims=True)
    plot, ax = plt.subplots()

    ax.plot(t, Vt_mean[0, :])
    ax.plot(t, 0.026 * np.ones_like(t))
    ax.set_xlabel('t')
    ax.set_ylabel('Vt')
    plt.title('Spot Variance')
    plt.show()


def calImpliedVol(ST, timespan):
    k = np.arange(-0.4, 0.4, 0.01)
    K = np.array([math.e ** i for i in k])[np.newaxis, :]
    call_prices = np.mean(np.maximum(ST[:, np.newaxis] - K, 0), axis=0)[:, np.newaxis]
    iv = []
    for i in range(len(k)):
        iv.append(implied_vol(1, K[0][i], timespan, call_prices[i]))

    plot, axes = plt.subplots()
    axes.plot(k, iv)
    axes.set_ylabel('implied volatility', fontsize=16)
    axes.set_xlabel('k', fontsize=16)
    plt.show()
    return iv


def call_price(S, K, T, sigma):
    # r = 0
    d1 = (np.log(S / K) + (0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    g = gn.GaussianNoiseSimulation()
    C = S * g.cdf(d1) - K * g.cdf(d2)
    return C


def implied_vol(S, K, T, C):
    def cp(sigma):
        return call_price(S, K, T, sigma) - C

    return brentq(cp, a=1e-9, b=10.0)

if __name__ == '__main__':
    N = 10000
    timespan = 1.0
    steps = 100

    interval = timespan / steps

    X, dW = VolterraProcess(N, steps, interval)
    VolterraProcessTest(X[:, 0:steps], timespan, interval, steps)

    VarianceP, SP, t = simulate_rBergomi_model(X, dW, timespan, interval, steps)
    VtTest(VarianceP, timespan, interval)

    # stock price
    utils.draw_n_paths(1, timespan + interval, interval, [SP[0]], 'stock price')

    iv = calImpliedVol(SP[:, -1], timespan)