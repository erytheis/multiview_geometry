from scipy.spatial.distance import cdist
import numpy as np
from ransac import *


class Experimentator():

    def __init__(self, matches):
        self.matches = matches

    def calculate_distance(self, matches):
        points_1 = matches[:, [0, 1]]
        points_2 = matches[:, [2, 3]]
        error = np.sqrt(np.sum(((points_1 - points_2) ** 2))) / len(matches)
        return error

    def run_grid_search(self):
        outlier_proportion = 0.2

        # Define grid
        sample_size_grid = [8, 9, 10]
        error_threshold_grid = np.linspace(0.05, 1, 10)
        number_of_accepted_points_grid = [50, 75, 100, 125, 150]

        final_results = []

        for sample_size in sample_size_grid:
            number_of_iterations = calculate_number_of_iterations(sample_size, outlier_proportion)
            error_threshold_results = []
            for error_threshold in error_threshold_grid:
                accepted_points_results = []
                for number_of_accepted_points in number_of_accepted_points_grid:
                    # Average the error after running the algorithm several times
                    times = 10
                    error = 0
                    for i in range(times):
                        _, best_matches = run_ransac(self.matches, number_of_iterations, sample_size, error_threshold,
                                                     number_of_accepted_points, speed_up = True)
                        error += self.calculate_distance(best_matches)

                    accepted_points_results.append(error / times)
                error_threshold_results.append(accepted_points_results)
            final_results.append(error_threshold_results)

        return final_results
