import numpy as np

import sys

sys.path.append('..')
import utils
import GaussianNoise.gaussian_noise_simulation as gn


class choleskyMethodfBM:
    """
    using Cholesky method to generate fBm
    """

    def generateFBm(self, timespan, interval, H):
        size = int(timespan / interval)
        gaussian_noise = gn.GaussianNoiseSimulation()
        Gn = gaussian_noise.generateNGn(size, 'box-muller')  # Gaussian noise

        gamma = [utils.gamma(H, k) for k in range(0, size)]

        fGn = []
        L = np.zeros((size, size))

        L[0][0] = gamma[0]  # 1
        fGn.append(Gn[0])

        for i in range(1, size):
            # compute every row of L(n)
            L[i][0] = gamma[i]

            for j in range(1, i):
                L[i][j] = (gamma[i - j] - L[i] @ L[j]) / L[j][j]

            L[i][i] = np.sqrt(gamma[0] - sum((L[i, 0:i]) ** 2))

            # new fGn
            fGn.append(Gn[:(i + 1)] @ L[i, :(i + 1)])

        # Generate fBM by accumulating fGn
        fBM = [fGn[0]]
        for i in range(1, size):
            fBM.append(fBM[i - 1] + fGn[i])

        # scale
        fBM = [i * (interval ** H) for i in fBM]

        return fBM


if __name__ == '__main__':
    cm = choleskyMethodfBM()
    nfBm = []
    for _ in range(0, 50):
        fBm = cm.generateFBm(10, 0.01, 0.8)
        nfBm.append(fBm)
    utils.draw_n_paths(50, 10, 0.01, nfBm)
