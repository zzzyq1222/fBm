import math
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append('..')
import GaussianNoise.gaussian_noise_simulation as gn

"""
    Euler-Maruyama method and Refined Euler method
"""

class EulerMethod:

    def generateDwt(self, n, steps):
        res = np.zeros([n, steps])
        gaussian_noise = gn.GaussianNoiseSimulation()
        for i in range(n):
            res[i, :] = gaussian_noise.generateNGn(steps, 'box-muller')
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
        dWt = self.generateDwt(n, steps)
        processes = np.zeros((n, steps))  # store each step

        yt = y0
        for j in range(n):  # generate n processes
            processes[j][0] = y0
            y = []
            for i in range(1, steps):
                yt = yt + theta * (mu - yt) * interval + \
                     sigma * math.sqrt(interval) * dWt[j][i]
                y.append(yt)
            processes[j][1:] = y
            yt = y0
        return processes

    def simulateGBM(self, mu, sigma, s0, timespan=10, interval=0.001, n=10):
        steps = int(timespan / interval)
        dWt = self.generateDwt(n, steps)
        processes = np.zeros((n, steps))  # store each step
        St = s0
        for j in range(n):  # generate n processes
            processes[j][0] = s0
            s = []
            for i in range(1, steps):
                St = St + mu * St * interval + \
                     sigma * St * math.sqrt(interval) * dWt[j][i]
                s.append(St)
            processes[j][1:] = s
            St = s0
        return processes

    """
    Refined Euler Method:
        1. Ornstein-Uhlenbeck process
            dY_t = theta*(mu-Y_t)dt + sigma*dW_t
            Y_0 = Y_init
            b' = 0
        2. GBM
            dS_t = mu*S_t*dt + sigma*S_t*dW_t
            b' = sigma
    """
    def simulateGBMRefined(self, mu, sigma, s0, timespan=10, interval=0.001, n=10):
        steps = int(timespan / interval)
        dWt = self.generateDwt(n, steps)
        processes = np.zeros((n, steps))  # store each step
        St = s0
        for j in range(n):  # generate n processes
            processes[j][0] = s0
            s = []
            for i in range(1, steps):
                St = St + mu * St * interval + \
                     sigma * St * math.sqrt(interval) * dWt[j][i] + \
                     0.5 * sigma * interval*(dWt[j][i]**2 - 1)
                s.append(St)
            processes[j][1:] = s
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
    results = euler_method.simulateOUProcess(0.7, 1.5, 0.06, 0, timespan=7, interval=0.001, n=10)
    # results = euler_method.simulateOUProcess(2, 0, 1, 1)
    euler_method.draw_paths(10, 7, 0.001, results)

    results = euler_method.simulateGBM(0, 1, 0.1)
    euler_method.draw_paths(10, 10, 0.001, results)

    results = euler_method.simulateGBMRefined(0, 1, 0.1)
    euler_method.draw_paths(10, 10, 0.001, results)

