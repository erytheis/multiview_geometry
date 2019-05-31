from experimentator import *
import matplotlib.pyplot as plt
import seaborn as sns

# Load matches
matches = np.loadtxt(
    '/Users/erytheis/PycharmProjects/Computer Vision/Assignment 3/description/data/library/library_matches.txt')

experimentator = Experimentator(matches)
results = experimentator.run_grid_search()

i = 1

for result in results:
    plt.subplot(1, 3, i)
    # plt.imshow(result, cmap='hot', interpolation='nearest')
    sns.heatmap(result, linewidth = 0.5)
    plt.title("Sample size = " + str(i + 7))
    i += 1
plt.show()

# TODO 1)create experimentator class and calculate the error measure (A * F.reshape(9,1))

