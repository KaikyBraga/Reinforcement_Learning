from sklearn.cluster import KMeans, DBSCAN
from  llm import LLMAgent

class Reviewer(LLMAgent):
    """ 
    This class represents the Reviewer. Evaluates clustering quality 
    based on evaluation metrics, suggests hyperparameter adjustments, 
    and checks computational efficiency.
    """

    def __init__(self, coder_instance, model="llama3.1"):
        super().__init__(model)  # Inicializa a classe LLMAgent
        self.coder = coder_instance
    
    def evaluate_accuracy(self):
        """Evaluate if clusters make sense based on an accuracy threshold (e.g., silhouette score > 0.5)"""

        silhouette_score, _ = self.coder.evaluate_clusters(metric="silhouette")
        
        # Geração de prompt para interação com o modelo
        if silhouette_score > 0.5:
            response = self.generate(f"The silhouette score is {silhouette_score}. The clusters are well separated.")
            return "Good cluster separation", response
        else:
            response = self.generate(f"The silhouette score is {silhouette_score}. The clusters need adjustments.")
            return "Consider adjusting parameters or algorithm", response


    def check_efficiency(self):
        """Analyze model efficiency and complexity"""

        if isinstance(self.coder.cluster_model, KMeans):
            # Analyzing the number of clusters chosen
            message = f"KMeans with {self.coder.cluster_model.n_clusters} clusters may have high computational cost if n_clusters is large."
            response = self.generate(message)

            return message, response
        
        elif isinstance(self.coder.cluster_model, DBSCAN):
            message = f"DBSCAN with eps={self.coder.cluster_model.eps} and min_samples={self.coder.cluster_model.min_samples} requires efficient data scaling."
            response = self.generate(message)

            return message, response


    # def suggest_hyperparameter_tuning(self):
    #     """Suggest hyperparameter changes for model improvement"""

    #     if isinstance(self.coder.cluster_model, KMeans):
    #         # Suggest incrementing clusters
    #         new_params = {"n_clusters": self.coder.cluster_model.n_clusters + 1}
    #         response = self.generate(f"Suggesting hyperparameter adjustment for KMeans: {new_params}")

    #         return new_params, response  
        
    #     elif isinstance(self.coder.cluster_model, DBSCAN):
    #         # Suggest reducing neighborhood radius for DBSCAN
    #         new_params = {"eps": self.coder.cluster_model.eps * 0.9}
    #         response = self.generate(f"Suggesting hyperparameter adjustment for DBSCAN: {new_params}")

    #         return new_params, response
    

    # def review_code_quality(self):
    #     """Ensure code quality, such as checking documentation and modularity"""

    #     if self.coder.cluster_model:
    #         response = self.generate("Ensure that the code is modular and well documented.")
    #         return "Code quality is acceptable", response
    #     else:
    #         response = self.generate("Ensure all steps are clearly documented and modularized.")
    #         return "Ensure all steps are clearly documented and modularized.", response
