import numpy as np
from numpy.linalg import svd
from helpers import *


class Camera():

    def __init__(self, camera_matrix):
        self.camera_matrix = camera_matrix


def load_data(data=0):
    if data == 0:
        data = "house"
    else:
        data = "library"

    camera_matrix_1 = np.loadtxt('../description/data/' + data + '/' + data + '1_camera.txt')
    camera_matrix_2 = np.loadtxt('../description/data/' + data + '/' + data + '2_camera.txt')
    matches = np.loadtxt('../description/data/' + data + '/' + data + '_matches.txt')

    return camera_matrix_1, camera_matrix_2, matches


# Loading data
camera_matrix_1, camera_matrix_2, matches = load_data(0)

# Construct A matrix
p1, p2, p3 = camera_matrix_1
p1_p, p2_p, p3_p = camera_matrix_2

objects_coordinates = np.zeros((len(matches), 4))
for i in range(len(matches)):
    A = np.multiply(matches[i].reshape((4, 1)), np.array([p3, p3, p3_p, p3_p])) - np.array([p1, p2, p1_p, p2_p])
    U, D, V = svd(A)
    object_coordinates = V[-1] / V[-1, -1]
    objects_coordinates[i] = object_coordinates

plot_data(np.array(objects_coordinates))
