import numpy as np
import matplotlib.pyplot as plt
from cyvlfeat import sift
from scipy.spatial.distance import cdist
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D


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

def find_matching_points(image1, image2, n_levels = 3, distance_threshold = 300):
    """
    :param image1 and image2 must be RGB images
    :param n_levels: number of scales
    :param distance_threshold: a threshold to accept a given match
    :return: two numpy lists, each with keypoints in [x,y]
    """

    # TODO
    '''
    Important note : you  might need to change the parameters (sift parameters) inside this function to
    have more or better matches
    '''
    matches_1 = []
    matches_2 = []
    image1 = np.array(image1.convert('L'))
    image2 = np.array(image2.convert('L'))
    '''
    Each column of keypoints is a feature frame and has the format [X;Y;S;TH], where X,Y is the (fractional) center of
    the frame, S is the scale and TH is the orientation (in radians).

    AND each column of features is the descriptor of the corresponding frame in F.
    A descriptor is a 128-dimensional vector of class UINT8
    '''
    keypoints_1, features_1 = sift.sift(image1, compute_descriptor = True, n_levels = n_levels)
    keypoints_2, features_2 = sift.sift(image2, compute_descriptor = True, n_levels = n_levels)
    pairwise_dist = cdist(features_1, features_2)  # len(features_1) * len(features_2)
    closest_1_to_2 = np.argmin(pairwise_dist, axis = 1)
    for i, idx in enumerate(closest_1_to_2):
        if pairwise_dist[i, idx] <= distance_threshold:
            matches_1.append([keypoints_1[i][1], keypoints_1[i][0]])
            matches_2.append([keypoints_2[idx][1], keypoints_2[idx][0]])
    return np.array(matches_1), np.array(matches_2)

def get_matches_notre_dame():
    basewidth = 500
    I1 = Image.open('../description/data/NotreDame/NotreDame1.jpg')
    wpercent = (basewidth / float(I1.size[0]))
    hsize = int((float(I1.size[1]) * float(wpercent)))
    I1 = I1.resize((basewidth, hsize), Image.ANTIALIAS)
    I2 = Image.open('../description/data/NotreDame/NotreDame2.jpg')
    wpercent = (basewidth / float(I2.size[0]))
    hsize = int((float(I2.size[1]) * float(wpercent)))
    I2 = I2.resize((basewidth, hsize), Image.ANTIALIAS)
    matchpoints1, matchpoints2 = find_matching_points(I1, I2, n_levels = 3, distance_threshold = 500)
    matches = np.hstack((matchpoints1, matchpoints2))
    return matches, I1, I2


def plot_data(data):
    """
    Plotting data point in 3-D
    :param data: 3-D data
    """
    x_values = data[:, 0]
    y_values = data[:, 1]
    z_values = data[:, 2]

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlim(-500, 500)
    ax.set_ylim(-500, 500)
    ax.set_zlim(-500, 500)
    ax.scatter(x_values, y_values, z_values)
    plt.show()