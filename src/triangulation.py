import numpy as np


class Camera():

    def __init__(self, f, p_x, p_y):
        self.f = f
        self.p_x = p_x
        self.p_y = p_y

    def compute_camera_matrix(self):
        self.calibration_matrix = np.array([[self.f, 0, 0],
                                            [0, self.f, 0],
                                            [0, 0, 1]])
        self.camera_matrix = np.dot()
