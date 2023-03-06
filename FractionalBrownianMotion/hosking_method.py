import math

import numpy as np

import sys

sys.path.append('..')
import utils
import GaussianNoise.gaussian_noise_simulation as gn


class hoskingMethodFBm:

    def generateFBm(self, timespan, interval, H):
        size = int(timespan / interval)
        gaussian_noise = gn.GaussianNoiseSimulation()
        Gn = np.array(gaussian_noise.generateNGn(size, 'box-muller'))  # Gaussian noise

        gamma = np.array([utils.gamma(H, k+1) for k in range(0, size)])  # the last element won't be used

        fGn = np.zeros(size)
        mu = np.zeros(size)
        sigma2 = np.zeros(size)  # square of sigma
        dn = np.zeros(size)

        # Initialize
        fGn[0] = Gn[0]
        mu[0] = gamma[0]*Gn[0]
        sigma2[0] = 1 - gamma[0]**2
        dn[0] = gamma[0]
        tau = [gamma[0]*dn[0]]

        for i in range(1, size-1):
            # sigma^2
            sigma2[i] = sigma2[i - 1] - (gamma[i + 1] - tau[i - 1]**2) / sigma2[i - 1]

            # d(n)
            phi_n = (gamma[i + 1] - tau[i - 1]) / sigma2[i - 1]
            dn = dn - phi_n * dn[::-1]
            dn[i] = phi_n

            fGn[i] = math.sqrt(sigma2[i]) * Gn[i] + mu[i - 1]

            mu[i] = dn @ Gn[::-1]

            F = np.diag(np.ones(i))
            F = np.rot90(F, -1)
            tau.append(gamma[0:i] @ F @ dn[0:i])

        # Generate fBM by accumulating fGn
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
        fBm = cm.generateFBm(10, 0.01, 0.7)
        nfBm.append(fBm)
    utils.draw_n_paths(10, 10, 0.01, nfBm)