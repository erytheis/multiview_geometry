from experimentator import *
from helpers import *


# Load matches
matches, _, __ = get_matches_notre_dame()

# Run the experiment
experimentator = Experimentator(matches)
experimentator.run_grid_search("a_f")
