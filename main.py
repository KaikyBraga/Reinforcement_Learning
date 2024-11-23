import numpy as np

from coder import Coder
from reviewer import Reviewer
from system import System
from utils import epsilon_greedy_decay, calculate_reward, update_q_value

data = np.random.rand(100, 2)  

# Create the Coder object
coder = Coder(data=data, evaluation_results_initial= {"silhouette_score": 1, "davies_bouldin_score": 0, "n_clusters": 1})

# Choose the algorithm
coder.choose_algorithm()

# Adjust the parameters 
coder.adjust_parameters()
coder.adjust_parameters()


