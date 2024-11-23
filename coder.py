from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler, RobustScaler, Normalizer
from llm import LLMAgent
import numpy as np
import pandas as pd

class Coder(LLMAgent):
    """ 
    This class represents the Coder. Performs the main pre-processing 
    steps, choosing the clustering algorithm, adjusting the model and 
    evaluating the clusters.
    """ 

    def __init__(self, data, evaluation_results_initial, model="llama3.1"):
        super().__init__(model)
        base_prompt = "You are a data scientist specialized in machine learning. Your task is to solve clustering problem. Return only the requested requirement, without additional explanations. Focus solely on the specifications provided in the prompt."
        self.history = [{"role": "user", "content": base_prompt}]
        self.data = data
        self.cluster_model = KMeans()
        self.algorithm_choice = "kmeans"  
        self.evaluation_results = evaluation_results_initial
        self.parameters_kmeans = {"n_clusters": 3}
        self.parameters_dbscan = {"eps": 0.5, "min_samples": 5}
        self.llm_error_flag = False # Flag to track LLM's delirium
        self.parameters_error_flag = False # Flag to track invallid parameters

    # Action 1
    def choose_algorithm(self):
        """Select clustering algorithm: kmeans or dbscan, based on provided metrics."""

        # Reset flags
        self.llm_error_flag = False  
        self.parameters_error_flag = False  
        
        prompt = (
            "Choose between the clustering algorithms 'kmeans' or 'dbscan'. "
            "Base your decision on the previous parameters and metrics: "
            f"Silhouette Score = {self.evaluation_results['silhouette_score']}, "
            f"Davies-Bouldin Score = {self.evaluation_results['davies_bouldin_score']}, "
            f"Number of Clusters = {self.evaluation_results.get('n_clusters', 'unknown')}. "
            "Select the algorithm that would potentially improve the clustering quality, "
            "maximizing the Silhouette Score and minimizing the Davies-Bouldin Score. "
            "RETURN ONLY THE NAME OF THE SELECTED ALGORITHM: 'kmeans' or 'dbscan'."
        )
        
        self.add_to_history({"role": "user", "content": prompt})
        response = self.generate(prompt)
        self.add_to_history({"role": "assistant", "content": response})

        print(response)

        try:
            if "kmeans" in response.lower():
                self.algorithm_choice = "kmeans"
                self.cluster_model = KMeans()

            elif "dbscan" in response.lower():
                self.algorithm_choice = "dbscan"
                self.cluster_model = DBSCAN()

            else:
                raise ValueError(f"Unexpected algorithm choice: {response}")
            
            if self.algorithm_choice == "kmeans":
                self.cluster_model.set_params(**self.parameters_kmeans)

            elif self.algorithm_choice == "dbscan":
                self.cluster_model.set_params(**self.parameters_dbscan)

            self.fit_model()
            self.evaluate_clusters() 

        except Exception as e:
            print(f"Error during algorithm selection: {e}")
            self.llm_error_flag = True  # Mark as error

        if self.evaluation_results["silhouette_score"] == None:
            self.parameters_error_flag = True  # Mark as error

    # Action 2
    def adjust_parameters(self):
        """Adjust parameters for the selected clustering algorithm."""

        if not self.algorithm_choice:
            raise ValueError("Algorithm has not been chosen yet.")
        
        # Reset flags
        self.llm_error_flag = False  
        self.parameters_error_flag = False  

        if self.algorithm_choice == "kmeans":
            prompt = (
                "Based on the previous results, optimize the `n_clusters` parameter "
                "for the KMeans algorithm. Use the provided metrics for decision-making: "
                f"Silhouette Score = {self.evaluation_results['silhouette_score']}, "
                f"Davies-Bouldin Score = {self.evaluation_results['davies_bouldin_score']}, "
                f"Number of Clusters = {self.evaluation_results.get('n_clusters', 'unknown')}. "
                "RETURN ONLY THE ADJUSTED PARAMETERS IN THE FORMAT: 'n_clusters=<new_value>'."
            )
        elif self.algorithm_choice == "dbscan":
            prompt = (
                "Based on the previous results, optimize the `eps` and `min_samples` parameters "
                "for the DBSCAN algorithm. Use the provided metrics for decision-making: "
                f"Silhouette Score = {self.evaluation_results['silhouette_score']}, "
                f"Davies-Bouldin Score = {self.evaluation_results['davies_bouldin_score']}, "
                f"Number of Clusters = {self.evaluation_results.get('n_clusters', 'unknown')}. "
                "RETURN ONLY THE ADJUSTED PARAMETERS IN THE FORMAT: 'eps=<new_value>, min_samples=<new_value>'."
            )
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm_choice}")
        
        self.add_to_history({"role": "user", "content": prompt})
        response = self.generate(prompt)
        self.add_to_history({"role": "assistant", "content": response})

        print(response)

        try:
            if response[0] == "'":
                response = response.strip("'")

            params = {}
            for param in response.split(","):
                key, value = param.split("=")
                params[key.strip()] = float(value.strip()) if "." in value else int(value.strip())
            self.cluster_model.set_params(**params)
            
            if self.algorithm_choice == "kmeans":
                self.cluster_model.set_params(**self.parameters_kmeans)

            elif self.algorithm_choice == "dbscan":
                self.cluster_model.set_params(**self.parameters_dbscan)

            self.fit_model() 
            self.evaluate_clusters() 

        except Exception as e:
            print(f"Error parsing parameters: {e}. Using default parameters.")
            self.llm_error_flag = True  # Mark as error

        if self.evaluation_results["silhouette_score"] == None:
            self.parameters_error_flag = True  # Mark as error


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
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        if n_clusters <= 1:
            self.evaluation_results = {
                "silhouette_score": None,
                "davies_bouldin_score": None,
                "n_clusters": n_clusters}
            return

        self.evaluation_results = {
            "silhouette_score": silhouette_score(self.data, labels),
            "davies_bouldin_score": davies_bouldin_score(self.data, labels),
            "n_clusters": n_clusters}
        
    def choose_norm(self):
        prompt = f"""Para poder fazer uma boa clusterização, é necessário que os dados estejam normalizados.
                  Seu dever é escolher o método que melhor se encaixe para normalizar o seguinte array de dados: {self.data}. 
                  Escolha entre os seguintes métodos: MaxAbsScaler, MinMaxScaler, StandardScaler, RobustScaler, Normalizer.
                  RESPONDA APENAS COM O NOME DO MÉTODO COMO LHE FOI PASSADO!"""

        
        self.add_to_history({"role": "user", "content": prompt})
        response = self.generate(prompt)
        self.add_to_history({"role": "assistant", "content": response})

        norm_names = ["MaxAbsScaler", "MinMaxScaler", "StandardScaler", "RobustScaler", "Normalizer"]
        list_norms = [MaxAbsScaler(), MinMaxScaler(), StandardScaler(), RobustScaler(), Normalizer()]
        for i in range(len(list_norms)):
            if norm_names[i] in response.lower():
                self.scaler = list_norms[i]
                return
        else:
            raise ValueError(f"Unsupported norm: {self.scaler}")
        
    def normalize_data(self):
        self.scaler.fit_transform(self.data)
