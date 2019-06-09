import numpy as np
from numpy.linalg import svd

from src.helpers import add_ones_column


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
    """
    Run RANSAC with a given set of parameters
    :param matches: (n x 4) matches found by SIFT alglorithm
    :param num_of_iterations: number of iterations
    :param sample_size: number of elements sampled in one iteration
    :param error_threshold: threshold of a good result
    :param num_of_accepted_points: the number of points at which the model is considered good
    :return: Fundamental matrix and the array of inliers
    """
    max_number_of_inliers, F_best, best_matches = 0, 0, 0
    sample_batch = matches

    for i in range(num_of_iterations):

        # Pick s points from mathces
        sample_idx = np.random.randint(0, sample_batch.shape[0], sample_size)
        sample = sample_batch[sample_idx]

        F = fit_fundamental_matrix(sample)

        # Compare the distance
        error = calculate_error(matches, F, method = 'sampson')

        # Filter the points within the band width
        inliers_idx = np.where(error < error_threshold)[0]
        inliers = matches[inliers_idx]

        # Save the best model
        if len(inliers) > max_number_of_inliers:
            max_number_of_inliers = len(inliers)
            F_best = F
            best_matches = inliers

        # If the number of inliers is bigger than the number of accepted points then this is a good model,
        # refit the model using all these inliers
        if  len(inliers) > num_of_accepted_points:
            sample_batch = inliers

    # print "Number of inliers " + str(max_number_of_inliers)
    return F_best, best_matches


def calculate_error(matches, F, method = 'a_f'):
    """
    A method that calculates the error of fundamental matrix with different methods
    :param matches: (n x 4) array of correspondenses
    :param F: (3 x 3) fundamental matrix
    :param method: {sampson, a_f, euclidean)
    :return: a vector of values of error
    """
    err = 0
    x = add_ones_column(matches[:, [0, 1]]).T
    x_p = add_ones_column(matches[:, [2, 3]]).T
    if method == 'sampson':
        F1 = np.dot(F, x)
        F2 = np.dot(F, x_p)
        denom = F1[0] ** 2 + F1[1] ** 2 + F2[0] ** 2 + F2[1] ** 2
        err = (np.diag(np.dot(x.T, np.dot(F, x_p)))) ** 2 / denom
    elif method == 'a_f':
        err = np.dot(x.T, np.dot(F, x_p))
    elif method == 'euclidean':
        err = np.sqrt(np.sum(((x - x_p) ** 2))) / len(matches)
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
    sample_size = 10
    outlier_proportion = 0.6
    number_of_iterations = calculate_number_of_iterations(sample_size, outlier_proportion)
    # number_of_iterations = 2000
    error_threshold = 0.08
    number_of_accepted_points = 150

    print "Expected number of iteration = " + str(number_of_iterations)
    return run_ransac(matches, number_of_iterations, sample_size, error_threshold, number_of_accepted_points)
