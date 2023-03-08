import math
import numpy as np
import sys

sys.path.append('..')
import utils
import GaussianNoise.gaussian_noise_simulation as gn


class hoskingMethodFBm:
    """
    using Hosking method to generate fBm
    Based on 2.2.1 of Simulation of fractional Brownian motion
    """

    def generateFBm(self, timespan, interval, H):
        size = int(timespan / interval)
        gaussian_noise = gn.GaussianNoiseSimulation()
        Gn = np.array(gaussian_noise.generateNGn(size, 'box-muller'))  # Gaussian noise
        gamma = np.array([utils.gamma(H, k + 1) for k in range(0, size + 1)])  # auto-covariance

        fGn = np.zeros(size)
        mu = np.zeros(size)
        sigma2 = np.zeros(size)  # square of sigma
        dn = np.zeros(size)
        tau = np.zeros(size)

        # Initialize

        fGn[0] = Gn[0]
        dn[0] = gamma[0]
        mu[0] = dn[0] * fGn[0]
        sigma2[0] = 1 - gamma[0] ** 2
        tau[0] = gamma[0] * dn[0]

        for i in range(1, size):
            # sigma^2
            sigma2[i] = sigma2[i - 1] - (gamma[i] - tau[i - 1]) ** 2 / sigma2[i - 1]

            # d(n)
            phi_n = (gamma[i] - tau[i - 1]) / sigma2[i - 1]

            F = np.diag(np.ones(i))
            F = np.rot90(F, -1)

            dn[:i] = dn[:i] - phi_n * np.dot(dn[:i], F)
            dn[i] = phi_n

            # fractional Gn
            mu[i] = np.dot(dn[:i], np.dot(fGn[:i], F))

            fGn[i] = math.sqrt(sigma2[i - 1]) * Gn[i] + mu[i - 1]

            F = np.diag(np.ones(i + 1))
            F = np.rot90(F, -1)
            tau[i] = np.dot(np.dot(gamma[:i + 1], F), dn[:(i + 1)])

        # Generate fBM by summing fGn
        fBM = [fGn[0]]
        for i in range(1, size):
            fBM.append(fBM[i - 1] + fGn[i])

        # scale
        fBM = [i * (interval ** H) for i in fBM]

        return fBM


if __name__ == '__main__':
    cm = hoskingMethodFBm()
    nfBm = []
    for _ in range(0, 10):
        fBm = cm.generateFBm(10, 0.01, 0.2)
        nfBm.append(fBm)
    utils.draw_n_paths(10, 10, 0.01, nfBm)
