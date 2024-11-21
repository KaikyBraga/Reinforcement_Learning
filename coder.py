from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from llm import LLMAgent
import numpy as np

class Coder(LLMAgent):
    """ 
    This class represents the Coder. Performs the main pre-processing 
    steps, choosing the clustering algorithm, adjusting the model and 
    evaluating the clusters.
    """ 

    def __init__(self, data, evaluation_results_initial, model="llama3.1"):
        super().__init__(model)
        self.base_prompt = "You are a data scientist specialized in machine learning. Your task is to solve clustering problem. Return only the requested requirement, without additional explanations. Focus solely on the specifications provided in the prompt."
        self.history = [{"role": "user", "content": self.base_prompt}]
        self.data = data
        self.cluster_model = KMeans()
        self.algorithm_choice = "kmeans"  
        self.evaluation_results = evaluation_results_initial
        self.llm_error_flag = False

    # Action 1
    def choose_algorithm(self):
        """Select clustering algorithm: kmeans or dbscan, based on provided metrics."""
        self.llm_error_flag = False  # Reset flag
        prompt = (
            "Choose between the clustering algorithms 'kmeans' or 'dbscan'. "
            "Base your decision on the previous parameters and metrics: "
            f"Silhouette Score = {self.evaluation_results['silhouette_score']}, "
            f"Davies-Bouldin Score = {self.evaluation_results['davies_bouldin_score']}. "
            "Select the algorithm that would potentially improve the clustering quality, "
            "maximizing the Silhouette Score and minimizing the Davies-Bouldin Score. "
            "RETURN ONLY THE NAME OF THE SELECTED ALGORITHM: 'kmeans' or 'dbscan'."
        )
        
        self.add_to_history({"role": "user", "content": prompt})
        response = self.generate(prompt)
        self.add_to_history({"role": "assistant", "content": response})

        print(response)

        try:
            if response.strip().lower() == "kmeans":
                self.algorithm_choice = "kmeans"
                self.cluster_model = KMeans()

            elif response.strip().lower() == "dbscan":
                self.algorithm_choice = "dbscan"
                self.cluster_model = DBSCAN()

            else:
                raise ValueError(f"Unexpected algorithm choice: {response}")
            
            self.fit_model()
            self.evaluate_clusters() 

        except Exception as e:
            print(f"Error during algorithm selection: {e}")
            self.llm_error_flag = True  # Mark as error


    def adjust_parameters(self):
        """Adjust parameters for the selected clustering algorithm."""

        if not self.algorithm_choice:
            raise ValueError("Algorithm has not been chosen yet.")
        
        self.llm_error_flag = False  # Reset flag

        if self.algorithm_choice == "kmeans":
            prompt = (
                "Based on the previous results, optimize the `n_clusters` parameter "
                "for the KMeans algorithm. Use the provided metrics for decision-making: "
                f"Silhouette Score = {self.evaluation_results['silhouette_score']}, "
                f"Davies-Bouldin Score = {self.evaluation_results['davies_bouldin_score']}. "
                "RETURN ONLY THE ADJUSTED PARAMETERS IN THE FORMAT: 'n_clusters=<new_value>'."
            )
        elif self.algorithm_choice == "dbscan":
            prompt = (
                "Based on the previous results, optimize the `eps` and `min_samples` parameters "
                "for the DBSCAN algorithm. Use the provided metrics for decision-making: "
                f"Silhouette Score = {self.evaluation_results['silhouette_score']}, "
                f"Davies-Bouldin Score = {self.evaluation_results['davies_bouldin_score']}. "
                "RETURN ONLY THE ADJUSTED PARAMETERS IN THE FORMAT: 'eps=<new_value>, min_samples=<new_value>'."
            )
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm_choice}")
        
        self.add_to_history({"role": "user", "content": prompt})
        response = self.generate(prompt)
        self.add_to_history({"role": "assistant", "content": response})

        print("RESPOSTA: ", response)

        try:
            params = {}
            for param in response.split(","):
                key, value = param.split("=")
                params[key.strip()] = float(value.strip()) if "." in value else int(value.strip())
            self.cluster_model.set_params(**params)
            
            self.fit_model()
            self.evaluate_clusters() 

        except Exception as e:
            print(f"Error parsing parameters: {e}. Using default parameters.")
            self.llm_error_flag = True  # Mark as error


    def fit_model(self):
        """Fit the clustering model to the preprocessed data."""

        if self.cluster_model is None:
            raise ValueError("Cluster model has not been initialized.")
        if self.data is None:
            raise ValueError("Data has not been provided.")
        
        try:
            self.cluster_model.fit(self.data)
        except Exception as e:
            print(f"Error during model fitting: {e}")
            raise

    def get_labels(self):
        """Get labels from the fitted clustering model."""
        
        if not hasattr(self.cluster_model, "labels_"):
            raise ValueError("Model has not been fitted yet.")
        return self.cluster_model.labels_


    def evaluate_clusters(self):
        """Evaluate clusters using Silhouette and Davies-Bouldin scores."""

        labels = self.get_labels()
        self.evaluation_results = {
            "silhouette_score": silhouette_score(self.data, labels),
            "davies_bouldin_score": davies_bouldin_score(self.data, labels)
        }

# Create some random data
data = np.random.rand(100, 2)  

# Create the Coder object
coder = Coder(data=data, evaluation_results_initial= {"silhouette_score": 1, "davies_bouldin_score": 0})

# Choose the algorithm
coder.choose_algorithm()

# Adjust the parameters 
coder.adjust_parameters()

coder.fit_model()

print(coder.evaluation_results)