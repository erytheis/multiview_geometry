import numpy as np
from numpy.linalg import svd
from helpers import *


def load_data(data = 0):
    """
    Load pictures from the /data/
    :param data: 0 - house photos, 1 - library
    :return: pictures and corresponding matches
    """
    if data == 0:
        data = "house"
    else:
        data = "library"

    camera_matrix_1 = np.loadtxt('../description/data/' + data + '/' + data + '1_camera.txt')
    camera_matrix_2 = np.loadtxt('../description/data/' + data + '/' + data + '2_camera.txt')
    matches = np.loadtxt('../description/data/' + data + '/' + data + '_matches.txt')

    return camera_matrix_1, camera_matrix_2, matches, data


def reconstruct_3d_points(camera_matrix_1, camera_matrix_2, matches):
    """ Reconstruct 3-D points from two pictures """
    p1, p2, p3 = camera_matrix_1
    p1_p, p2_p, p3_p = camera_matrix_2

    objects_coordinates = np.zeros((len(matches), 4))
    for i in range(len(matches)):
        A = np.multiply(matches[i].reshape((4, 1)), np.array([p3, p3, p3_p, p3_p])) - np.array([p1, p2, p1_p, p2_p])
        U, D, V = svd(A)
        object_coordinates = V[-1] / V[-1, -1]
        objects_coordinates[i] = object_coordinates

    return objects_coordinates


def calculate_camera_center(camera_matrix):
    U, D, V = svd(camera_matrix)
    camera_center = V[-1] / V[-1, -1]
    return camera_center


""" Main algorithm """
for i in [0, 1]:
    camera_matrix_1, camera_matrix_2, matches, title = load_data(i)
    camera1_coordinates = calculate_camera_center(camera_matrix_1)
    camera2_coordinates = calculate_camera_center(camera_matrix_2)
    coordinates = reconstruct_3d_points(camera_matrix_1, camera_matrix_2, matches)
    data_dict = {"Camera 1": np.array([camera1_coordinates]), "Camera 2": np.array([camera2_coordinates]),
                 "Points": coordinates}
    plot_data(title, **data_dict)
