import matplotlib.pyplot as plt
import seaborn as sns
from helpers import plot_error_distributions
from ransac import *


def plot_heat_map(error_type, error_threshold_grid, final_results, number_of_accepted_points_grid, sample_size_grid):
    for grid_pos, result in enumerate(final_results):
        plt.subplot(1, 3, grid_pos + 1)
        sns.heatmap(result,
                    vmin = np.min(final_results), vmax = np.max(final_results),
                    xticklabels = number_of_accepted_points_grid, yticklabels = np.round(error_threshold_grid, 2),
                    linewidth = 0.5, annot = True)
        plt.xlabel("T = N accepted points")
        plt.ylabel("d = Error threshold")

        plt.title("Sample size = " + str(sample_size_grid[grid_pos]))

    plt.suptitle(error_type + " error")
    plt.show()


def calculate_distance(matches):
    points_1 = matches[:, [0, 1]]
    points_2 = matches[:, [2, 3]]
    error = np.sqrt(np.sum(((points_1 - points_2) ** 2))) / len(matches)
    return error


# def calculate_error_helper(matches, F):
#     keypoints = add_ones_column(matches[:, [0, 1]])
#     ground_truth = add_ones_column(matches[:, [2, 3]])
#     return np.sum(calculate_error(keypoints, F, ground_truth, method='sampson')) / len(matches)


class Experimentator:

    def __init__(self, matches):
        self.matches = matches

    def run_grid_search(self, error_type = "algebraic_distance", plot = True):
        outlier_proportion = 0.2

        # Define grid
        sample_size_grid = [8, 9, 10]
        error_threshold_grid = np.linspace(0.01, 0.05, 10)
        number_of_accepted_points_grid = [50, 75, 100, 125, 150]
        times = 10

        final_results = []


        for grid_pos, sample_size in enumerate(sample_size_grid):
            number_of_iterations = calculate_number_of_iterations(sample_size, outlier_proportion)
            error_threshold_results = []

            print "Started on " + str(grid_pos + 1) + " iteration of sample grid size..."
            for error_threshold in error_threshold_grid:
                accepted_points_results = []
                for number_of_accepted_points in number_of_accepted_points_grid:
                    # Average the error after running the algorithm several times
                    error = 0
                    number_of_accepted_points_history = []

                    for i in range(times):

                        F, best_matches = run_ransac(self.matches, number_of_iterations, sample_size, error_threshold,
                                                     number_of_accepted_points, speed_up = True)
                        error += np.average( calculate_error(best_matches, F, method = error_type))


                        number_of_accepted_points_history.append(len(best_matches))

                    accepted_points_results.append(error / times)
                error_threshold_results.append(accepted_points_results)

            final_results.append(error_threshold_results)

        if plot:
            plot_heat_map(error_type, error_threshold_grid, final_results, number_of_accepted_points_grid,
                          sample_size_grid)
        return final_results
