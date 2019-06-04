import numpy as np
from numpy.linalg import svd
from helpers import *

class Camera():

    def __init__(self, camera_matrix):
        self.camera_matrix = camera_matrix


# Loading data
camera_matrix_1 = np.loadtxt('../description/data/library/library1_camera.txt')
camera_matrix_2 = np.loadtxt('../description/data/library/library2_camera.txt')
matches = np.loadtxt('../description/data/library/library_matches.txt')

# Construct A matrix
p1, p2, p3 = camera_matrix_1
p1_p, p2_p, p3_p = camera_matrix_2

objects_coordinates = np.zeros((len(matches),4))
for i in range(len(matches)):
    A = np.multiply(matches[i].reshape((4, 1)), np.array([p3, p3, p3_p, p3_p])) - np.array([p1, p2, p1_p, p2_p]).T
    U, D, V = svd(A)
    object_coordinates = V[-1] / V[-1, -1]
    objects_coordinates[i] = object_coordinates


plot_data(np.array(objects_coordinates))
