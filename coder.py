from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from llm import *

class Coder(LLMAgent):
    """ 
    This class represents the Coder. Performs the main pre-processing 
    steps, choosing the clustering algorithm, adjusting the model and 
    evaluating the clusters.
    """ 

    def __init__(self, data, model="llama3.1", problem_description=""):
        super().__init__(model)
        self.base_prompt = ("You are a Python developer and data scientist. Your job is to write code to solve data-science problems. "
                            "Be concise and make sure to document your code.")
        self.problem_description = problem_description
        self.data = data
        self.model = None
        # Default algorithm
        self.algorithm_choice = "kmeans"  

    
    def choose_algorithm(self, algorithm="kmeans", **kwargs):
        """Select clustering algorithm: kmeans or dbscan, with specified kwargs"""

        # Store the algorithm choice
        self.algorithm_choice = algorithm
        
        # Generate a prompt for the LLM about the choice of algorithm
        prompt = f"The chosen algorithm for clustering is {algorithm}. Adjusting parameters accordingly."
        response = self.generate(prompt)

        if algorithm == "kmeans":
            self.model = KMeans(**kwargs)
        elif algorithm == "dbscan":
            self.model = DBSCAN(**kwargs)
        else:
            raise ValueError("Unsupported algorithm selected.")
        
        return response
    
    
    def fit_model(self):
        """Fit the clustering model to the preprocessed data"""

        if self.model and self.data is not None:
            self.model.fit(self.data)
        else:
            raise ValueError("Model or data not initialized.")
        
        # After fitting the model, inform the LLM about the next step
        response = self.generate("Clustering model has been fitted. Proceeding to evaluate the results.")
        return response

    
    def get_labels(self):
        """Get labels of the fitted model"""
        if hasattr(self.model, "labels_"):
            return self.model.labels_
        else:
            raise ValueError("Model has not been fitted yet.")
    

    def evaluate_clusters(self, metric="silhouette"):
        """Evaluate clusters using silhouette score or davies_bouldin"""

        labels = self.get_labels()

        # Evaluate the clusters based on the selected metric
        if metric == "silhouette":
            score = silhouette_score(self.data, labels)
        elif metric == "davies_bouldin":
            score = davies_bouldin_score(self.data, labels)
        else:
            raise ValueError("Unsupported metric selected.")
        
        # Generate a prompt for LLM to report the evaluation
        prompt = f"Clustering evaluation complete using the {metric} score. The score is: {score}."
        response = self.generate(prompt)
        
        return score, response
