import numpy as np
import pandas as pd
import warnings 

warnings.filterwarnings("ignore")

from coder import Coder
from reviewer import Reviewer
from system import System
from utils import epsilon_greedy_decay, calculate_reward, update_q_value

# TESTE
np.random.seed(42)
data = np.random.randn(200, 5)  

initial_metrics = {
    "silhouette_score": 0.3,  
    "davies_bouldin_score": 2.0,  
    "n_clusters": 3
}

df = pd.DataFrame(data, columns=["feature1", "feature2", "feature3", "feature4", "feature5"])

coder = Coder(data=data, df=df, evaluation_results_initial=initial_metrics)
reviewer = Reviewer(coder)

sistema = System(coder, reviewer)

sistema.train(epochs=10, epsilon=1.0, epsilon_min=0.01, decay_rate=0.99, size_penalty=0.5, lambda_k=0.5, lambda_size=0.5)