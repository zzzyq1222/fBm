import math
import numpy
"""
Generate Gaussian noise
1. Inverse transforming
2. Box-Muller method
"""

"""
Inverse transforming
Input: number of points
"""


class GaussianNoiseSimulation:
    def __init__(self):
        pass

    """
    using the method given by 26.2.17 formula in 'Handbook of 
    Mathematical Functions With Formulas, Graphs, and Mathematical Tables'
    """

    # pdf of N(0,1)
    def pdf(self, x) -> float:
        pdf = (1 / math.sqrt(2 * math.pi)) * (math.e ** (-(x ** 2) / 2))
        return pdf

    def cdf(self, x, sigma=1, mu=0):
        x_norm = (x - mu) / sigma
        p = 0.2316419
        b1 = 0.319381530
        b2 = -0.356563782
        b3 = 1.781477937
        b4 = -1.821255978
        b5 = 1.330274429
        t = 1 / (p * x_norm + 1)
        P = 1 - self.pdf(x_norm) * t * (b1 + t * (b2 + t * (b3 + t * (b4 + t * b5))))
        return round(P, 8)

    """z is the generated uniform variable"""

    # todo 调参数
    def inverse(self, z, lower=-8.0, upper=8.0, e=1e-8) -> float:
        mid = lower + (upper - lower) / 2
        cdf = self.cdf(mid)
        if math.fabs(cdf-z) <= e:
            return mid
        if cdf >= z:
            return self.inverse(z, lower, mid)
        else:
            return self.inverse(z, mid, upper)

    """generate n uniform var and computes the corresponding normal var"""
    def generateNGn(self, n):
        u = numpy.random.uniform(0,1,n)
        generated_gn = []
        for i in range(n):
            generated_gn.append(self.inverse(u[i]))
        return generated_gn

# todo test

if __name__ == '__main__':
    gn = GaussianNoiseSimulation()
    # print("x = 0, sigma = 1, mu = 0: ", gn.cdf(0))
    # print("x = 0, sigma = 5, mu = 2: ", gn.cdf(0, 5, 2))
    # t = gn.inverse(0.1)
    # print("inverse of 0.1: ", t)
    n = 1000 # number of normal variables
    print(gn.generateNGn(n))

