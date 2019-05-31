import numpy as np
from numpy.linalg import svd
from scipy.spatial.distance import cdist
from helpers import *

# Load matches
matches = np.loadtxt(
    '/Users/erytheis/PycharmProjects/Computer Vision/Assignment 3/description/data/library/library_matches.txt')


def fit_fundamental_matrix(matches):
    """
    Find the values of of fundamental matrix by running eight-point algorithm.
    :param matches: (m x 4) matrix , where m is the number of points
    :return: (3 x 3) matrix with fundamental matrix
    """
    # Construct matrix Mx9
    num_of_points = matches.shape[0]
    A = np.zeros((num_of_points, 9))

    # Construct lines for A
    for key, match in enumerate(matches):
        x, y, x_p, y_p = match
        A[key] = np.array([x * x_p, x * y_p, x, x_p * y, y * y_p, y, x_p, y_p, 1])

    # Take the last row and transpose == take the last column of transposed
    u, sigma, v_T = svd(A)
    F = v_T[-1].reshape((3, 3))

    # Enforce 2nd rank for F
    u_f, sigma_f, v_T_f = svd(F)
    sigma_f[-1] = 0
    F = np.dot(u_f, np.dot(np.diag(sigma_f), v_T_f))

    return F


def run_ransac(matches, num_of_iterations, sample_size, error_threshold, num_of_accepted_points):
    max_number_of_inliers = 0
    F_best = 0
    best_matches = 0
    for i in range(num_of_iterations):

        # Pick s points from mathces
        sample_idx = np.random.randint(0, matches.shape[0], sample_size)
        sample = matches[sample_idx]
        F = fit_fundamental_matrix(sample)

        # Take the remaining after sampling
        remaining_points = np.delete(matches, sample_idx, 0)

        keypoints = add_ones_column(remaining_points[:, [0, 1]])
        ground_truth = add_ones_column(remaining_points[:, [2, 3]])

        # Compare the distance
        error = calculate_distance(keypoints, F, ground_truth, method = 'sampson')

        # Filter the points within the band width
        inlier_idx = np.where(error < error_threshold)[0]
        inliers = matches[inlier_idx]

        # Save the best model
        if len(inlier_idx) > max_number_of_inliers:
            max_number_of_inliers = len(inlier_idx)
            F_best = F
            best_matches = inliers

    print "Number of inliers " + str(max_number_of_inliers)
    return F_best, best_matches


def calculate_distance(x, F, x_p, method = 'sampson'):
    err = 0
    x = x.T
    x_p = x_p.T
    if method == 'sampson':
        F1 = np.dot(F, x)
        F2 = np.dot(F, x_p)
        denom = F1[0] ** 2 + F1[1] ** 2 + F2[0] ** 2 + F2[1] ** 2
        err = (np.diag(np.dot(x.T, np.dot(F, x_p)))) ** 2 / denom
    elif method == 'algebraic_distance':
        err = np.dot(x.T, np.dot(F, x_p))
    return err


def calculate_number_of_iterations(s, e, p = 0.99):
    """
    Calculate minimum number of required samplings in order to achieve not contaminated set of points
    :param s: number of samples in one sampling
    :param e: proportion of outliers,
    :param p: Probability that at least one of our samples of s points consists just of inliers
    :return: the minimum number of samplings (rounded to the higher number)
    """
    assert e > 0
    return int(np.ceil(np.log(1 - p) / np.log((1 - np.power((1 - e), s)))))


def build_A(matches):
    """
    Construct (m x 9) matrix, where m is the number of points
    :param matches: array of (m x 4) of [x, y, x', y']
    :return: (m x 9) matrix A
    """
    num_of_points = matches.shape[0]
    A = np.zeros((num_of_points, 9))
    for key, match in enumerate(matches):
        x, y, x_p, y_p = match
        A[key] = np.array([x * x_p, x * y_p, x, x_p * y, y * y_p, y, x_p, y_p, 1])
    return A


def RANSAC_for_fundamental_matrix(matches):
    # Hyperparameters
    sample_size = 8
    outlier_proportion = 0.3
    number_of_iterations = 3 * calculate_number_of_iterations(sample_size, outlier_proportion)
    error_threshold = 0.8
    number_of_accepted_points = 20
    print "Expected number of iteration = " + str(number_of_iterations)
    return run_ransac(matches, number_of_iterations, sample_size, error_threshold, number_of_accepted_points)

# TODO 1)create experimentator class and calculate the error measure (A * F.reshape(9,1))
