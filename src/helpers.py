import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from cyvlfeat import sift
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist
from scipy.interpolate import griddata


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


def plot_data(title="", **data):
    """
    Plotting data point in 3-D
    :param data: 3-D data
    """
    fig = plt.figure()
    ax = Axes3D(fig)

    box_limit_max = np.max([np.max(value) for value in data.values()])
    box_limit_min = np.min([np.min(value) for value in data.values()])

    ax.set_xlim(box_limit_min, box_limit_max)
    ax.set_ylim(box_limit_min, box_limit_max)
    ax.set_zlim(box_limit_min, box_limit_max)
    # ax.scatter(data[:, 0], data[:, 1], data[:, 2])

    for key, points in data.items():
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], label=key, s=[5 if key == "Points" else 20])
    plt.legend()
    plt.title(title)
    plt.show()


def plot_data_interpolated(title, method = 'cubic', **data):
    """
    Interpolates the points in 3-d to create a mesh
    :param data: (M-D) dataset
    :param method: {"linear", nearest, "cubic"}
    """
    fig = plt.figure()
    ax = Axes3D(fig)

    # box_limit_max = np.max(data[:, 0:1])
    # box_limit_min = np.min(data[:, 0:1])

    box_limit_max = np.max([np.max(value) for value in data.values()])
    box_limit_min = np.min([np.min(value) for value in data.values()])

    points = np.array([data["Points"][:, 0], data["Points"][:, 1]]).T
    values = data["Points"][:, 2]
    grid_x, grid_y = np.mgrid[box_limit_min:box_limit_max:100j, box_limit_min:box_limit_max:100j]
    Z = griddata(points, values, (grid_x, grid_y), method = method)

    ax.set_xlim(box_limit_min, box_limit_max)
    ax.set_ylim(box_limit_min, box_limit_max)
    ax.set_zlim(box_limit_min, box_limit_max)


    for key, points in data.items():
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], label=key, s=[5 if key == "Points" else 20])
    ax.plot_surface(grid_x, grid_y, Z, alpha = 0.25, color = "g")

    plt.title(title)
    plt.show()
