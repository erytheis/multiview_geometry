import numpy as np
import matplotlib.pyplot as plt

def add_ones_column(points):
    """
    Add a column of ones to the right side of the point matrix
    :param points:
    :return:
    """
    result = np.ones((points.shape[0], points.shape[1] + 1))
    result[:, [0, 1]] = points
    return result


def plot_error_distributions(errors):
    """
    Get the distribution of errors for visualization
    :param errors: vector of errors calculated in ransac
    :return:
    """
    n, bins, patches = plt.hist(x = errors, bins = "auto", color = '#0504aa',
                                alpha = 0.7, rwidth = 0.85)
    maxfreq = n.max()
    plt.ylim(ymax = np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.show()

