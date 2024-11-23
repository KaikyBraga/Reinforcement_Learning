import numpy as np
import pandas as pd

from coder import Coder
from reviewer import Reviewer
from system import System
from utils import epsilon_greedy_decay, calculate_reward, update_q_value

# TESTE

np.random.seed(42)
data = np.random.randn(200, 5)  
print(np.max(data), np.min(data))

initial_metrics = {
    "silhouette_score": 0.3,  
    "davies_bouldin_score": 2.0,  
    "n_clusters": 3
}

coder = Coder(data=data, evaluation_results_initial=initial_metrics)

print("\nTestando escolha de método de normalização...")
coder.choose_norm()
coder.normalize_data()
print("\nDados normalizados:")
print(np.max(data), np.min(data))

# escolha do algoritmo
print("\nTestando escolha do algoritmo...")
coder.choose_algorithm()
print(f"Algoritmo escolhido: {coder.algorithm_choice}")

# ajuste de parâmetros
print("\nTestando ajuste de parâmetros...")
coder.adjust_parameters()
print(f"Parâmetros ajustados para {coder.algorithm_choice}: {coder.cluster_model.get_params()}")

# remoção de outliers
print("\nTestando remoção de outliers...")
print(f"Antes da remoção: {len(coder.df)} amostras")
coder.remove_outliers()
print(f"Depois da remoção: {len(coder.df)} amostras")

print(coder.evaluation_results)
