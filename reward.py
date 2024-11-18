from sklearn.metrics import silhouette_score, davies_bouldin_score
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans


def calcular_recompensa(X, labels_anteriores, labels_novos, lambda_k=0.1, lambda_tamanho=0.5, t_min=5):
    """
    Calcula a recompensa para um agente de clusterização.

    Parâmetros:
    - X: array-like, shape (n_samples, n_features)
        Dados utilizados para calcular as métricas.
    - labels_anteriores: array-like, shape (n_samples,)
        Rótulos de cluster anteriores à ação.
    - labels_novos: array-like, shape (n_samples,)
        Rótulos de cluster após a ação.
    - lambda_k: float
        Penalização para número excessivo de clusters.
    - lambda_tamanho: float
        Penalização para clusters com poucos elementos.
    - t_min: int
        Tamanho mínimo esperado para clusters.

    Retorno:
    - recompensa: float
        Recompensa calculada.
    """

    # Cálculo das métricas
    silhouette_anterior = silhouette_score(X, labels_anteriores) if len(set(labels_anteriores)) > 1 else 0
    silhouette_novo = silhouette_score(X, labels_novos) if len(set(labels_novos)) > 1 else 0
    
    davies_bouldin_anterior = davies_bouldin_score(X, labels_anteriores) if len(set(labels_anteriores)) > 1 else np.inf
    davies_bouldin_novo = davies_bouldin_score(X, labels_novos) if len(set(labels_novos)) > 1 else np.inf

    # Recompensa base (melhoria nas métricas)
    recompensa_metrica = (silhouette_novo - silhouette_anterior) - (davies_bouldin_novo - davies_bouldin_anterior)

    # Penalização por número de clusters
    k_novo = len(set(labels_novos))
    penalizacao_k = lambda_k * k_novo

    # Penalização por clusters pequenos
    tamanhos_clusters = [sum(labels_novos == cluster) for cluster in set(labels_novos)]
    penalizacao_tamanho = lambda_tamanho * sum(1 for tamanho in tamanhos_clusters if tamanho < t_min)

    # Recompensa final
    recompensa = recompensa_metrica - penalizacao_k - penalizacao_tamanho

    return recompensa

# Dados
X, _ = make_blobs(n_samples=1000, centers=4, random_state=42, cluster_std=0.7)

# Clusterização antes da ação
kmeans_anterior = KMeans(n_clusters=3, random_state=42)
labels_anteriores = kmeans_anterior.fit_predict(X)

# Clusterização após a ação
kmeans_novo = KMeans(n_clusters=4, random_state=42)
labels_novos = kmeans_novo.fit_predict(X)

# Cálculo da recompensa
recompensa = calcular_recompensa(X, labels_anteriores, labels_novos)
print(f"Recompensa: {recompensa:.4f}")
