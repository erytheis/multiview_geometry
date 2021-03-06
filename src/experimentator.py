import matplotlib.pyplot as plt

from ransac import *
from src.helpers import plot_heat_map



class Experimentator:

    def __init__(self, matches):
        self.matches = matches

    def run_grid_search(self, error_type="sampson", plot=True):
        """
        Study the effect in the error of the error threshold and number of accepted points
        for different samples sizes

        """

        outlier_proportion = 0.3

        # Define grid
        sample_size_grid = [8, 9, 10]
        error_threshold_grid = np.round(np.linspace(0.002, 0.2, 10), 3)
        number_of_accepted_points_grid = [80, 100, 120, 140, 160, 180]
        times = 10

        final_results_errors = []
        final_results_inliers = []

        for grid_pos, sample_size in enumerate(sample_size_grid):
            number_of_iterations = calculate_number_of_iterations(sample_size, outlier_proportion)
            print "Started iteration {0} of {1}. Sample size is now {2}.".format(grid_pos + 1, len(sample_size_grid),
                                                                                 sample_size)

            number_of_inliers = []
            errors_res = []
            # print "Started on " + str(grid_pos + 1) + " iteration of sample size " + " ..."
            for number_of_accepted_points in number_of_accepted_points_grid:
                errors = []
                inliers = []
                for error_threshold in error_threshold_grid:
                    # Average the error after running the algorithm several times
                    error, number_of_inliers_accepted = self.execute_ransac(error_threshold, error_type,
                                                                            number_of_accepted_points,
                                                                            number_of_iterations, sample_size,
                                                                            times)

                    errors.append(error)
                    inliers.append(number_of_inliers_accepted)

                number_of_inliers.append(inliers)
                errors_res.append(errors)
                print str(number_of_accepted_points) + " accepted points finished"

            final_results_inliers.append(number_of_inliers)
            final_results_errors.append(errors_res)

        if plot:
            axis = {"y_label": "T = Number of accepted points",
                    "x_label": "d = Error threshold",
                    "y_tick_labels": number_of_accepted_points_grid,
                    "x_tick_labels": error_threshold_grid,
                    "sub_title": ["Sample size = " + str(sample_size) for sample_size in sample_size_grid],
                    "title": error_type + " error",
                    "results": np.array(final_results_errors)}
            # plot_heat_map(axis, True)
            plot_heat_map(axis, False)

            axis["title"] = "Number of inliers"
            axis["results"] = np.array(final_results_inliers)
            # plot_heat_map(axis, True)
            plot_heat_map(axis, False)
            plt.show()

    def execute_ransac(self, error_threshold, error_type, number_of_accepted_points, number_of_iterations, sample_size,
                       times):
        """ Execute ransac 'times' times and return the average of the result """
        error = 0
        number_of_accepted_matches_history = []
        for i in range(times):
            F, best_matches = run_ransac(self.matches, number_of_iterations, sample_size, error_threshold,
                                         number_of_accepted_points)
            error += np.average(calculate_error(best_matches, F, method=error_type))

            number_of_accepted_matches_history.append(len(best_matches))
        return np.average(error), int(np.average(number_of_accepted_matches_history))

    def study_inliers_search(self):
        """
        Study the effect in the number of inliers that the outlier proportion and error threshold have
        for different sample sizes
        """

        sample_size_grid = [8, 9, 10]
        outlier_proportion_grid = np.round(np.linspace(0.1, 0.5, 6), 2)
        error_threshold_grid = np.round(np.linspace(0.002, 1, 10), 4)
        number_of_accepted_points = 100
        times = 10

        error_type = "a_f"
        final_results = []

        for i, sample_size in enumerate(sample_size_grid):
            print "Started sample size {0}, iteration {1} of {2}.".format(sample_size, i + 1, len(sample_size_grid))

            outliers_results = []
            for grid_pos, outlier_proportion in enumerate(outlier_proportion_grid):
                number_of_iterations = calculate_number_of_iterations(sample_size, outlier_proportion)
                if number_of_iterations > 5000:
                    number_of_iterations = 5000
                print "Started iteration {0} of {1}. Outlier proportion is now {2}, will do {3} itertations in RANSAC" \
                    .format(grid_pos + 1, len(outlier_proportion_grid), outlier_proportion, number_of_iterations)
                error_threshold_results = []

                for error_threshold in error_threshold_grid:
                    errors, number_of_accepted_matches = self.execute_ransac(error_threshold, error_type,
                                                                             number_of_accepted_points,
                                                                             number_of_iterations, sample_size, times)
                    error_threshold_results.append(number_of_accepted_matches[0])

                outliers_results.append(error_threshold_results)
            final_results.append(outliers_results)

        axis = {"x_label": "d = Error threshold",
                "y_label": "Outlier proportion",
                "x_tick_labels": error_threshold_grid,
                "y_tick_labels": outlier_proportion_grid,
                "sub_title": ["Sample size = " + str(sample_size) for sample_size in sample_size_grid],
                "title": error_type + " error. Number of accepted matches.",
                "results": final_results}
        plot_heat_map(axis, True)
        plot_heat_map(axis, False)
        plt.show()
