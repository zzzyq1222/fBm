import math
import numpy as np
import sys
sys.path.append('..')
import GaussianNoise.gaussian_noise_simulation as gn

class BMRandomWalk:
    """using the property of independent increments of BM to construct random walk"""

    def generateBM(self, time, interval):
        gaussian_noise = gn.GaussianNoiseSimulation()
        size = int(time / interval)
        data = gaussian_noise.generateNGn(size, 'box-muller')
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

# if __name__ == '__main__':
#     bmr = BMRandomWalk()
#     paths = 100
#     interval = 0.0001
#     timespan = 5
#     n_paths = bmr.generateNBM(paths, timespan, interval)  # data
#     utils.draw_bm_paths(paths, timespan, interval, n_paths)
