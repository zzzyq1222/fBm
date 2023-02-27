import math
import numpy
import matplotlib
import matplotlib.pyplot as plt

"""
Generate Gaussian noise
1. Inverse transforming
2. Box-Muller method
"""


class GaussianNoiseSimulation:
    def __init__(self):
        pass

    """
    1. Inverse transforming method
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
        if x_norm >= 0:
            t = 1 / (p * x_norm + 1)
            P = 1 - self.pdf(x_norm) * t * (b1 + t * (b2 + t * (b3 + t * (b4 + t * b5))))
            # P = 1-self.pdf(x_norm)*(b1*t+b2*(t**2)+b3*(t**3)+b4*(t**4)+b5*(t**5))
        else:
            t = 1 / (-p * x_norm + 1)
            P = self.pdf(x_norm) * t * (b1 + t * (b2 + t * (b3 + t * (b4 + t * b5))))
        return P

    """z is the generated uniform variable"""

    def inverse(self, z, lower=-8.0, upper=8.0, e= 1e-8) -> float:
        mid = lower + (upper - lower) / 2
        cdf = self.cdf(mid)
        if math.fabs(cdf - z) <= e:
            return mid
        if cdf >= z:
            return self.inverse(z, lower, mid)
        else:
            return self.inverse(z, mid, upper)

    def inverse_transform_method(self, n):
        u = numpy.random.uniform(0, 1, n)
        generated_gn = []
        for i in range(n):
            generated_gn.append(self.inverse(u[i]))
        return generated_gn

    """generate n uniform var and computes the corresponding normal var"""

    """
    2. Box-Muller method
    input: n
    output: 2*n normal random numbers
    """

    def box_muller_method(self, n):
        u1 = numpy.random.uniform(0, 1, n)
        u2 = numpy.random.uniform(0, 1, n)
        generated_gn = []
        for i in range(n):
            n1 = math.sqrt(-2 * math.log(u1[i])) * math.cos(2 * math.pi * u2[i])
            n2 = math.sqrt(-2 * math.log(u1[i])) * math.sin(2 * math.pi * u2[i])
            generated_gn.extend([n1, n2])

        return generated_gn

    def generateNGn(self, n, method):
        if method == 'inverse':
            return self.inverse_transform_method(n)
        elif method == 'box-muller':
            return self.box_muller_method(n)[0:n]

    def draw_histogram(self, data, name):
        matplotlib.rcParams['axes.unicode_minus'] = False
        plt.hist(data, bins=100, density=True, facecolor="blue", edgecolor="black")
        plt.xlabel("x")
        plt.ylabel("probability")
        plt.title("Distribution using " + name)
        plt.show()

    def cal_sigma(self, data, mu):
        sum = 0
        for i in range(len(data)):
            try:
                sum += math.pow(data[i] - mu, 2)
            except:
                print(data[i], mu)
                # given the hint by Numerical solution of SDE
        return sum / (len(data) - 1)

    def cal_mu(self, data):
        s = sum(data)
        return s / len(data)


if __name__ == '__main__':
    gn = GaussianNoiseSimulation()
    # print("x = 0, sigma = 1, mu = 0: ", gn.cdf(0))
    # print("x = 0, sigma = 5, mu = 2: ", gn.cdf(0, 5, 2))
    # t = gn.inverse(0.1)
    # print("inverse of 0.1: ", t)
    num = 1_000_000  # number of normal variables
    data_inverse = gn.generateNGn(num, 'inverse')
    sigma_inverse = gn.cal_sigma(data_inverse, 0)
    mu_inverse = gn.cal_mu(data_inverse)
    gn.draw_histogram(data_inverse, "inverse transforming method")

    data_box_muller = gn.generateNGn(num, 'box-muller')
    sigma_box_muller = gn.cal_sigma(data_box_muller, 0)
    mu_box_muller = gn.cal_mu(data_box_muller)
    gn.draw_histogram(data_box_muller, "Box-Muller method")

    print("Inverse method: sigma = ", sigma_inverse,", mu = ", mu_inverse)
    print("Box-Muller method: sigma = ", sigma_box_muller, ", mu = ", mu_box_muller)
