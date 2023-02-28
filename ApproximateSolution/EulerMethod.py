import math
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append('..')
import GaussianNoise.gaussian_noise_simulation as gn


class EulerMethod:
    """
    use Euler-Maruyama method to approximate the solution to
    1. Ornstein-Uhlenbeck process
        dY_t = theta*(mu-Y_t)dt + sigma*dW_t
        Y_0 = Y_init
    2. GBM
    """

    def generateDwt(self, n, steps):
        res = np.zeros([n, steps])
        gaussian_noise = gn.GaussianNoiseSimulation()
        for i in range(n):
            res[i, :] = gaussian_noise.generateNGn(steps, 'box-muller')
        return res

    def simulateOUProcess(self, theta, mu, sigma, y0, timespan=10, interval=0.001, n=10):
        steps = int(timespan / interval)
        dWt = self.generateDwt(n, steps)
        processes = np.zeros((n, steps))  # store each step

        yt = y0
        for j in range(n):
            processes[j][0] = y0
            y = []
            for i in range(1, steps):
                yt = yt + theta * (mu - yt) * interval + \
                     sigma * math.sqrt(interval) * dWt[j][i]
                y.append(yt)

            processes[j][1:] = y
            yt = 0
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
    results = euler_method.simulateOUProcess(0.7, 1.5, 0.06, 0)

    euler_method.draw_paths(10, 10, 0.001, results)
