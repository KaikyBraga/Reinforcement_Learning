from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler


class Coder:
    """ 
    This class represents the Coder. Performs the main pre-processing 
    steps, choosing the clustering algorithm, adjusting the model and 
    evaluating the clusters.
    """ 

    def __init__(self, data):
        self.data = data
        self.model = None
        self.scaled_data = None

    
    def preprocess_data(self, scale=True):
        """Preprocess data by scaling it if scale=True"""

        if scale:
            scaler = StandardScaler()
            self.scaled_data = scaler.fit_transform(self.data)
        else:
            self.scaled_data = self.data

    
    def choose_algorithm(self, algorithm="kmeans", **kwargs):
        """Select clustering algorithm: kmeans or dbscan, with specified kwargs"""

        if algorithm == "kmeans":
            self.model = KMeans(**kwargs)
        elif algorithm == "dbscan":
            self.model = DBSCAN(**kwargs)
        else:
            raise ValueError("Unsupported algorithm selected.")
        
    
    def fit_model(self):
        """Fit the clustering model to the preprocessed data"""

        if self.model and self.scaled_data is not None:
            self.model.fit(self.scaled_data)
        else:
            raise ValueError("Model or data not initialized.")
    

    def get_labels(self):
        """Get labels of the fitted model"""
        if hasattr(self.model, "labels_"):
            return self.model.labels_
        else:
            raise ValueError("Model has not been fitted yet.")
        

    def evaluate_clusters(self, metric="silhouette"):
        """Evaluate clusters using silhouette score or davies_bouldin"""

        labels = self.get_labels()
        if metric == "silhouette":
            return silhouette_score(self.scaled_data, labels)
        elif metric == "davies_bouldin":
            return davies_bouldin_score(self.scaled_data, labels)
        else:
            raise ValueError("Unsupported metric selected.")
