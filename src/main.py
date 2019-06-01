from experimentator import *

# Load matches
matches = np.loadtxt(
    '../description/data/library/library_matches.txt')
    # '/Users/erytheis/PycharmProjects/Computer Vision/Assignment 3/description/data/library/library_matches.txt')

experimentator = Experimentator(matches)
results = experimentator.run_grid_search()

# TODO 1)create experimentator class and calculate the error measure (A * F.reshape(9,1))
