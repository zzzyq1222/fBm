import math
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append('..')
# import GaussianNoise.gaussian_noise_simulation as gn
import BrownianMotion.random_walk as rw
import utils

"""
    Euler-Maruyama method and Refined Euler method
"""


class EulerMethod:

    def generateWt(self, n, steps):
        res = np.zeros([n, steps])
        brownian_motion = rw.BMRandomWalk()
        for i in range(n):
            res[i, :] = brownian_motion.generateBM(steps, 1)
        return res

    """
    use Euler-Maruyama method to approximate the solution to
    1. Ornstein-Uhlenbeck process
        dY_t = theta*(mu-Y_t)dt + sigma*dW_t
    2. GBM
        dS_t = mu*S_t*dt + sigma*S_t*dW_t

    """

    def simulateOUProcess(self, theta, mu, sigma, y0, timespan=10, interval=0.001, n=10):
        steps = int(timespan / interval)
        Wt = self.generateWt(n, steps)
        processes = np.zeros((n, steps))  # store each step

        yt = y0
        for j in range(n):  # generate n processes
            y = [y0]
            for i in range(1, steps):
                yt = yt + theta * (mu - yt) * interval + \
                     sigma * math.sqrt(interval) * (Wt[j][i]-Wt[j][i-1])
                y.append(yt)
            processes[j][0:] = y
            yt = y0
        return processes

    def simulateGBM(self, mu, sigma, s0, timespan=10, interval=0.001, n=10):
        steps = int(timespan / interval)
        Wt = self.generateWt(n, steps)
        processes = np.zeros((n, steps))  # store each step
        St = s0
        for j in range(n):  # generate n processes
            s = [s0]
            for i in range(1, steps):
                St = St + mu * St * interval + \
                     sigma * St * math.sqrt(interval) * (Wt[j][i]-Wt[j][i-1])
                s.append(St)
            processes[j][0:] = s
            St = s0
        return processes

    """
    Refined Euler Method:
        1.GBM
            dS_t = mu*S_t*dt + sigma*S_t*dW_t
            b' = sigma
    """

    def simulateGBMRefined(self, mu, sigma, s0, timespan=10, interval=0.001, n=10):
        steps = int(timespan / interval)
        Wt = self.generateWt(n, steps)
        processes = np.zeros((n, steps))  # store each step
        St = s0
        for j in range(n):  # generate n processes
            s = [s0]
            for i in range(1, steps):
                St = St + mu * St * interval + \
                     sigma * St * math.sqrt(interval) * (Wt[j][i]-Wt[j][i-1]) + \
                     0.5 * sigma * interval * ((Wt[j][i]-Wt[j][i-1]) ** 2 - 1)
                s.append(St)
            processes[j][0:] = s
            St = s0
        return processes

    def draw_paths(self, paths, timespan, interval, data):
        fig, ax = plt.subplots()
        x = np.linspace(0, timespan, int(timespan / interval))
        for i in range(paths):
            y = data[i]
            ax.plot(x, y, linewidth=0.5)
        plt.show()


if __name__ == '__main__':
    euler_method = EulerMethod()

    #
    results = euler_method.simulateOUProcess(0.7, 1.5, 0.06, 0, timespan=10, interval=0.001, n=10)
    # results = euler_method.simulateOUProcess(2, 0, 1, 1)
    utils.draw_n_paths(10, 10, 0.001, results)

    results = euler_method.simulateGBM(0.5, 0.5, 10)
    utils.draw_n_paths(10, 10, 0.001, results)

    results = euler_method.simulateGBMRefined(0.5, 0.5, 10)
    utils.draw_n_paths(10, 10, 0.001, results)
