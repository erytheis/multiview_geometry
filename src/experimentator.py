import matplotlib.pyplot as plt
import seaborn as sns

from ransac import *


def plot_heatmap(error_threshold_grid, error_threshold_results, i, number_of_accepted_points_grid, sample_size):
    plt.subplot(1, 3, i)
    sns.heatmap(error_threshold_results,
                vmin=np.min(error_threshold_results), vmax=np.max(error_threshold_results),
                xticklabels=number_of_accepted_points_grid, yticklabels=np.round(error_threshold_grid, 2),
                linewidth=0.5)
    plt.title("Sample size = " + str(sample_size))


def calculate_distance(matches):
    points_1 = matches[:, [0, 1]]
    points_2 = matches[:, [2, 3]]
    error = np.sqrt(np.sum(((points_1 - points_2) ** 2))) / len(matches)
    return error


def calculate_error_helper(matches, F):
    keypoints = add_ones_column(matches[:, [0, 1]])
    ground_truth = add_ones_column(matches[:, [2, 3]])
    return np.sum(calculate_error(keypoints, F, ground_truth, method='sampson')) / len(matches)


class Experimentator:

    def __init__(self, matches):
        self.matches = matches

    def run_grid_search(self, error_type="epipolar", plot=True):
        outlier_proportion = 0.2

        # Define grid
        sample_size_grid = [8, 9, 10]
        error_threshold_grid = np.linspace(0.05, 1, 10)
        number_of_accepted_points_grid = [50, 75, 100, 125, 150]

        final_results = []

        for grid_pos, sample_size in enumerate(sample_size_grid):
            number_of_iterations = calculate_number_of_iterations(sample_size, outlier_proportion)
            error_threshold_results = []
            for error_threshold in error_threshold_grid:
                accepted_points_results = []
                for number_of_accepted_points in number_of_accepted_points_grid:
                    # Average the error after running the algorithm several times
                    times = 100
                    error = 0
                    for i in range(times):
                        F, best_matches = run_ransac(self.matches, number_of_iterations, sample_size, error_threshold,
                                                     number_of_accepted_points, speed_up=True)
                        if error_type == "euclidean":
                            error += calculate_distance(best_matches)
                        else:
                            error += calculate_error_helper(best_matches, F)

                    accepted_points_results.append(error / times)
                error_threshold_results.append(accepted_points_results)

            if plot:
                plot_heatmap(error_threshold_grid, error_threshold_results, grid_pos + 1,
                             number_of_accepted_points_grid, sample_size)

            final_results.append(error_threshold_results)

        if plot:
            plt.show()
        return final_results