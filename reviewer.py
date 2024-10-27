from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import numpy as np


class Reviewer:
    """ 
    This class represents the Reviewer. Evaluates clustering quality 
    based on evaluation metrics, suggests hyperparameter adjustments, 
    and checks computational efficiency.
    """

    def __init__(self, coder_instance):
        self.coder = coder_instance
    

    def evaluate_accuracy(self):
        """Evaluate if clusters make sense based on an accuracy threshold (e.g., silhouette score > 0.5)"""

        silhouette = self.coder.evaluate_clusters(metric="silhouette")
        if silhouette > 0.5:
            return "Good cluster separation"
        else:
            return "Consider adjusting parameters or algorithm"
    

    def suggest_hyperparameter_tuning(self):
        """Suggest hyperparameter changes for model improvement"""

        if isinstance(self.coder.model, KMeans):
            # Suggest incrementing clusters
            return {"n_clusters": self.coder.model.n_clusters + 1}  
        elif isinstance(self.coder.model, DBSCAN):
            # Suggest reducing neighborhood radius for DBSCAN
            return {"eps": self.coder.model.eps * 0.9}  
    

    def check_efficiency(self):
        """Analyze model efficiency and complexity"""

        if isinstance(self.coder.model, KMeans):
            # Analyzing the number of clusters chosen
            return f"KMeans with {self.coder.model.n_clusters} clusters may have high computational cost if n_clusters is large."
        elif isinstance(self.coder.model, DBSCAN):
            return f"DBSCAN with eps={self.coder.model.eps} and min_samples={self.coder.model.min_samples} requires efficient data scaling."


    def review_code_quality(self):
        """Ensure code quality, such as checking documentation and modularity"""

        return "Code quality is acceptable" if self.coder.model else "Ensure all steps are clearly documented and modularized."
    