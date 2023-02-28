import matplotlib.pyplot as plt
import numpy as np


def draw_n_paths(paths, timespan, interval, data):
    fig, ax = plt.subplots()
    x = np.linspace(0, timespan, int(timespan / interval))
    for i in range(paths):
        y = data[i]
        ax.plot(x, y, linewidth=0.5)
    plt.show()
