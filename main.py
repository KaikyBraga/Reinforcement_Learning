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

df = pd.DataFrame(data, columns=["feature1", "feature2", "feature3", "feature4", "feature5"])

coder = Coder(data=data, df=df, evaluation_results_initial=initial_metrics)
reviewer = Reviewer(coder)

sistema = System(coder, reviewer)

sistema.train(epochs=10, epsilon=1.0, epsilon_min=0.01, decay_rate=0.99, size_penalty=0.5, lambda_k=0.5, lambda_size=0.5)

### PASSO 1

# # escolha do algoritmo
# print("\nTestando escolha do algoritmo...")
# coder.choose_algorithm()
# print(f"Algoritmo escolhido: {coder.algorithm_choice}")

# # ajuste de parâmetros
# print("\nTestando ajuste de parâmetros...")
# coder.adjust_parameters()
# print(f"Parâmetros ajustados para {coder.algorithm_choice}: {coder.cluster_model.get_params()}")

# results = list()
# results.append(coder.evaluation_results)

# # PASSO 2

# # remoção de outliers
# print("\nTestando remoção de outliers...")
# print(f"Antes da remoção: {len(coder.df)} amostras")
# coder.remove_outliers()
# print(f"Depois da remoção: {len(coder.df)} amostras")

# # escolha do algoritmo
# print("\nTestando escolha do algoritmo...")
# coder.choose_algorithm()
# print(f"Algoritmo escolhido: {coder.algorithm_choice}")

# # ajuste de parâmetros
# print("\nTestando ajuste de parâmetros...")
# coder.adjust_parameters()
# print(f"Parâmetros ajustados para {coder.algorithm_choice}: {coder.cluster_model.get_params()}")

# results.append(coder.evaluation_results)

# ### PASSO 3

# print("\nTestando escolha de método de normalização...")
# coder.choose_norm()
# coder.normalize_data()
# print("\nDados normalizados:")
# print(np.max(coder.data), np.min(coder.data))

# # escolha do algoritmo
# print("\nTestando escolha do algoritmo...")
# coder.choose_algorithm()
# print(f"Algoritmo escolhido: {coder.algorithm_choice}")

# # ajuste de parâmetros
# print("\nTestando ajuste de parâmetros...")
# coder.adjust_parameters()
# print(f"Parâmetros ajustados para {coder.algorithm_choice}: {coder.cluster_model.get_params()}")

# results.append(coder.evaluation_results)

# print(results)