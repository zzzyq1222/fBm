import math
import numpy as np
import sys
sys.path.append('..')
import GaussianNoise.gaussian_noise_simulation as gn


class BMCholeskyFac:
    def generateCovMatrix(self, interval, n):
        cov_matrix = math.sqrt(interval) * np.ones([n, n])
        cov_matrix = np.tril(cov_matrix, k=0)
        return cov_matrix

    def generateBM(self, timespan, interval):
        gaussian_noise = gn.GaussianNoiseSimulation()
        size = int(timespan / interval)  # number of steps
        data = gaussian_noise.generateNGn(size, 'box-muller')  # normal variable
        cov_matrix = self.generateCovMatrix(interval, size)
        path = np.matmul(cov_matrix, data)
        return path

    def generateNBM(self, n, timespan, interval):
        size = int(timespan / interval)
        paths = np.zeros((n, size))
        for i in range(n):
            paths[i, :] = self.generateBM(timespan, interval)
        return paths

# if __name__ == '__main__':
#     bmcf = BMCholeskyFac()
#     paths = 100
#     interval = 0.001
#     timespan = 5
#     utils.draw_bm_paths(paths, timespan, interval, bmcf.generateNBM(paths, timespan, interval))
