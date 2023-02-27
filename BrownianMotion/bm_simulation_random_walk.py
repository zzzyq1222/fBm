import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import GaussianNoise.gaussian_noise_simulation as GN


class BMRandomWalk:
    """using the property of independent increments of BM to construct random walk"""

    def generateBM(self, time, interval):
        gn = GN.GaussianNoiseSimulation()
        size = int(time / interval)
        data = gn.generateNGn(size, 'box-muller')
        bm = [math.sqrt(interval) * data[0]]
        for i in range(1, size):
            bm.append(math.sqrt(interval) * data[i] + bm[i - 1])
        return bm

    # n is the number of paths
    def generateNBM(self, n, timespan, interval):
        size = int(timespan / interval)
        paths = np.zeros((n, size))
        for i in range(n):
            paths[i, :] = self.generateBM(timespan, interval)
        return paths

    def draw_bm_paths(self, paths, timespan, interval, data):
        fig, ax = plt.subplots()
        x = np.linspace(0, timespan, int(timespan / interval))
        y = []
        for i in range(paths):
            y = data[i]
            ax.plot(x, y, linewidth=0.5)
        plt.show()


if __name__ == '__main__':
    bmr = BMRandomWalk()
    paths = 100
    interval = 0.0001
    timespan = 5
    n_paths = bmr.generateNBM(paths, timespan, interval)  # data
    bmr.draw_bm_paths(paths, timespan, interval, n_paths)
