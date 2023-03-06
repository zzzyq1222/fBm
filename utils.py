import matplotlib.pyplot as plt
import numpy as np
import math

def draw_n_paths(paths, timespan, interval, data):
    fig, ax = plt.subplots()
    x = np.linspace(0, timespan, int(timespan / interval))
    for i in range(paths):
        y = data[i]
        ax.plot(x, y, linewidth=0.5)
    plt.show()


"""compute the auto-variance with H and interval k"""
def gamma(H,k):
    g = 0.5*(abs(k-1)**(2*H) - 2*abs(k)**(2*H) + abs(k+1)**(2*H))
    return g