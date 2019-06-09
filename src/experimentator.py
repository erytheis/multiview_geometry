import matplotlib.pyplot as plt
import seaborn as sns

from ransac import *


def plot_heat_map(axis, annot):
    """
    Plot heat map with the parameters sent
    :param axis: dictionary with the information for the labels. keys are:
                    x_tick_labels, y_tick_labels, x_label, y_label, results, title
    :param annot: flag to show the heat map annotated or not
    :return:
    """
    final_results = axis['results']
    for grid_pos, result in enumerate(final_results):
        plt.subplot(1, 3, grid_pos + 1)
        sns.heatmap(result,
                    vmin=np.min(final_results), vmax=np.max(final_results),
                    xticklabels=axis['x_tick_labels'], yticklabels=np.round(axis['y_tick_labels'], 4),
                    linewidth=0.5, annot=annot)
        plt.xlabel(axis['x_label'])
        plt.ylabel(axis['y_label'])
        plt.title(axis["sub_title"][grid_pos])

    plt.suptitle(axis['title'])
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


    def run_grid_search(self, error_type="algebraic_distance", plot=True):
        outlier_proportion = 0.3

        # Define grid
        sample_size_grid = [8, 9, 10]
        error_threshold_grid = np.linspace(0.002, 0.2, 20)
        number_of_accepted_points_grid = [50, 75, 100, 125, 150]
        times = 20

        final_results = []

        for grid_pos, sample_size in enumerate(sample_size_grid):
            number_of_iterations = calculate_number_of_iterations(sample_size, outlier_proportion)
            print "Started iteration {0} of {1}. Sample size is now {2}.".format(grid_pos + 1, len(sample_size_grid),
                                                                                 sample_size)
            error_threshold_results = []

            # print "Started on " + str(grid_pos + 1) + " iteration of sample size " + " ..."
            for error_threshold in error_threshold_grid:
                accepted_points_results, _ = self.run_accepted_points_search(error_threshold, error_type,
                                                                             number_of_accepted_points_grid,
                                                                             number_of_iterations, sample_size, times)

                error_threshold_results.append(accepted_points_results)
                print "Error threshold " + str(error_threshold) + " finished"

            final_results.append(error_threshold_results)

        if plot:
            axis = {"x_label": "T = N accepted points",
                    "y_label": "d = Error threshold",
                    "x_tick_labels": number_of_accepted_points_grid,
                    "y_tick_labels": error_threshold_grid,
                    "sub_title": ["Sample size = " + str(sample_size) for sample_size in sample_size_grid],
                    "title": error_type + " error",
                    "results": final_results}
            plot_heat_map(axis, True)
            plot_heat_map(axis, False)
        return final_results


    def run_accepted_points_search(self, error_threshold, error_type, number_of_accepted_points_grid,
                                   number_of_iterations, sample_size, times, use_T=True):
        errors = []
        number_of_accepted_matches = []
        for number_of_accepted_points in number_of_accepted_points_grid:
            # Average the error after running the algorithm several times
            error = 0
            number_of_accepted_matches_history = []

            for i in range(times):
                F, best_matches = run_ransac(self.matches, number_of_iterations, sample_size, error_threshold,
                                             number_of_accepted_points, use_T)
                error += np.average(calculate_error(best_matches, F, method=error_type))

                number_of_accepted_matches_history.append(len(best_matches))

            errors.append(error / times)
            number_of_accepted_matches.append(int(np.average(number_of_accepted_matches_history)))

        return errors, number_of_accepted_matches


    def study_accepted_points_search(self):
        sample_size_grid = [8, 9, 10]
        outlier_proportion_grid = np.round(np.linspace(0.1, 0.7, 6), 2)
        error_threshold_grid = np.round(np.linspace(0.002, 1, 10), 4)
        number_of_accepted_points_grid = [100]
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
                    errors, number_of_accepted_matches = self.run_accepted_points_search(error_threshold, error_type,
                                                                                         number_of_accepted_points_grid,
                                                                                         number_of_iterations,
                                                                                         sample_size,
                                                                                         times, False)
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

        pass

        # plt.plot(number_of_accepted_matches, label="Using T")
        # plt.plot(number_of_accepted_matches_no_T, label="Not using T")
        # plt.xlabel("T")
        # plt.ylabel("Number of accepted matches")
        # plt.legend()
        # plt.figure()
        #
        # plt.plot(errors, label="Using T")
        # plt.plot(errors_no_T, label="Not using T")
        # plt.xlabel("T")
        # plt.ylabel("Error")
        # plt.legend()
        # plt.show()

        # fig, ax1 = plt.subplots()
        #
        # color = 'tab:red'
        # ax1.set_xlabel('T, number of accepted points')
        # ax1.set_ylabel('Number of accepted matches', color=color)
        # ax1.plot(number_of_accepted_matches, label="Using T")
        # ax1.plot(number_of_accepted_matches_no_T, color=color, label="Not using T")
        # ax1.tick_params(axis='y', labelcolor=color)
        # plt.legend()
        #
        # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        #
        # color = 'tab:blue'
        # ax2.set_ylabel('Error', color=color)  # we already handled the x-label with ax1
        # ax2.plot(errors, color=color, label="Using T")
        # ax2.plot(errors_no_T, color=color, label="Not using T")
        # ax2.tick_params(axis='y', labelcolor=color)
        #
        # fig.tight_layout()  # otherwise the right y-label is slightly clipped
        # plt.legend()
        # plt.title("Effect of T in RANSAC performance")
        # plt.show()
