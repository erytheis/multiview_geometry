from experimentator import *
from helpers import *

matches, _, __ = get_matches_notre_dame()
# Load matches
# matches = np.loadtxt(
    # '../description/data/library/library_matches.txt')
    # '/Users/erytheis/PycharmProjects/Computer Vision/Assignment 3/description/data/library/library_matches.txt')
    # )

experimentator = Experimentator(matches)
experimentator.run_grid_search("a_f")
# experimentator.run_grid_search("euclidean")
# experimentator.run_grid_search()
# experimentator.study_accepted_points_search()
