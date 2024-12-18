from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, MeanShift
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler, RobustScaler, Normalizer
from llm import LLMAgent
import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

class Coder(LLMAgent):
    """ 
    This class represents the Coder. Performs the main pre-processing 
    steps, choosing the clustering algorithm, adjusting the model and 
    evaluating the clusters.
    """ 

    def __init__(self, data, df, evaluation_results_initial, model="llama3.1"):
        super().__init__(model)
        base_prompt = "You are a data scientist specialized in machine learning. Your task is to solve clustering problem. Return only the requested requirement, without additional explanations. Focus solely on the specifications provided in the prompt."
        self.history = [{"role": "user", "content": base_prompt}]
        self.df = df  # Pandas DataFrame
        self.data = data  # Array of df
        self.__backup_data = data
        self.__backup_df = df
        self.cluster_model = KMeans()  # Default Clustering Model
        self.algorithm_choice = "kmeans"
        self.evaluation_results = evaluation_results_initial
        self.parameters_kmeans = {"n_clusters": 3}
        self.parameters_dbscan = {"eps": 0.5, "min_samples": 5}
        self.parameters_agglomerative = {"n_clusters": 3}
        self.parameters_gmm = {"n_components": 3}
        self.parameters_meanshift = {}  # MeanShift does not have n_clusters, it is based on bandwidth
        self.llm_error_flag = False  # Flag to track LLM's delirium
        self.parameters_error_flag = False  # Flag to track invalid parameters
        self.n_cluster_invalid_flag = False  # Flag to track invalid n_clusters


    # Action 1
    def choose_algorithm(self):
        """Select clustering algorithm, based on provided metrics."""

        self.reset_flags()

        prompt = (
            "Choose between the clustering algorithms 'kmeans', 'dbscan', 'agglomerative', or 'meanshift'. "
            "Base your decision on the previous parameters and metrics: "
            f"Silhouette Score = {self.evaluation_results['silhouette_score']}, "
            f"Davies-Bouldin Score = {self.evaluation_results['davies_bouldin_score']}, "
            f"Number of Clusters = {self.evaluation_results.get('n_clusters', 'unknown')}. "
            "Select the algorithm that would potentially improve the clustering quality, "
            "maximizing the Silhouette Score and minimizing the Davies-Bouldin Score. "
            "RETURN ONLY THE NAME OF THE SELECTED ALGORITHM: 'kmeans', 'dbscan', 'agglomerative', or 'meanshift'."
        )

        self.add_to_history({"role": "user", "content": prompt})
        response = self.generate(prompt)
        self.add_to_history({"role": "assistant", "content": response})

        print("Response:", response)

        try:
            if "kmeans" in response.lower():
                self.algorithm_choice = "kmeans"
                self.cluster_model = KMeans()

            elif "dbscan" in response.lower():
                self.algorithm_choice = "dbscan"
                self.cluster_model = DBSCAN()

            elif "agglomerative" in response.lower():
                self.algorithm_choice = "agglomerative"
                self.cluster_model = AgglomerativeClustering()

            elif "gmm" in response.lower():
                self.algorithm_choice = "gmm"
                self.cluster_model = GaussianMixture()

            elif "meanshift" in response.lower():
                self.algorithm_choice = "meanshift"
                self.cluster_model = MeanShift()

            else:
                raise ValueError(f"Unexpected algorithm choice: {response}")

            # Set the parameters based on selected algorithm
            if self.algorithm_choice == "kmeans":
                self.cluster_model.set_params(**self.parameters_kmeans)

            elif self.algorithm_choice == "dbscan":
                self.cluster_model.set_params(**self.parameters_dbscan)

            elif self.algorithm_choice == "agglomerative":
                self.cluster_model.set_params(**self.parameters_agglomerative)

            elif self.algorithm_choice == "gmm":
                self.cluster_model.set_params(**self.parameters_gmm)

            elif self.algorithm_choice == "meanshift":
                self.cluster_model.set_params()

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
        
        self.reset_flags() 

        if self.algorithm_choice == "kmeans":
            prompt = (
                "Optimize the `n_clusters` parameter for the KMeans algorithm based on the following metrics: "
                f"Silhouette Score = {self.evaluation_results['silhouette_score']}, "
                f"Davies-Bouldin Score = {self.evaluation_results['davies_bouldin_score']}, "
                f"Number of Clusters = {self.evaluation_results.get('n_clusters', 'unknown')}. "
                "Provide a single integer value for `n_clusters` in the format: '<new_value>'."
            )
        elif self.algorithm_choice == "dbscan":
            prompt = (
                "Optimize the `eps` and `min_samples` parameters for the DBSCAN algorithm based on the following metrics: "
                f"Silhouette Score = {self.evaluation_results['silhouette_score']}, "
                f"Davies-Bouldin Score = {self.evaluation_results['davies_bouldin_score']}, "
                f"Number of Clusters = {self.evaluation_results.get('n_clusters', 'unknown')}. "
                "Provide two numeric values separated by a comma in the format: '<eps_value>, <min_samples_value>'."
            )
        elif self.algorithm_choice == "agglomerative":
            prompt = (
                "Optimize the `n_clusters` parameter for the AgglomerativeClustering algorithm based on the following metrics: "
                f"Silhouette Score = {self.evaluation_results['silhouette_score']}, "
                f"Davies-Bouldin Score = {self.evaluation_results['davies_bouldin_score']}, "
                f"Number of Clusters = {self.evaluation_results.get('n_clusters', 'unknown')}. "
                "Provide a single integer value for `n_clusters` in the format: '<new_value>'."
            )
        elif self.algorithm_choice == "gmm":
            prompt = (
                "Optimize the `n_components` parameter for the GaussianMixtureModel algorithm based on the following metrics: "
                f"Silhouette Score = {self.evaluation_results['silhouette_score']}, "
                f"Davies-Bouldin Score = {self.evaluation_results['davies_bouldin_score']}, "
                f"Number of Clusters = {self.evaluation_results.get('n_clusters', 'unknown')}. "
                "Provide a single integer value for `n_components` in the format: '<new_value>'."
            )
        elif self.algorithm_choice == "meanshift":
            prompt = (
                "Optimize the `bandwidth` parameter for the MeanShift algorithm based on the following metrics: "
                f"Silhouette Score = {self.evaluation_results['silhouette_score']}, "
                f"Davies-Bouldin Score = {self.evaluation_results['davies_bouldin_score']}, "
                f"Number of Clusters = {self.evaluation_results.get('n_clusters', 'unknown')}. "
                "Provide a single numeric value for `bandwidth` in the format: '<new_value>'."
            )
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm_choice}")
        
        self.add_to_history({"role": "user", "content": prompt})
        response = self.generate(prompt)
        self.add_to_history({"role": "assistant", "content": response})

        print("Response:", response)

        try:
            if response[0] == "'":
                response = response.strip("'")

            params = {}
            if self.algorithm_choice == "dbscan":
                eps, min_samples = map(float, response.split(","))
                params["eps"] = eps
                params["min_samples"] = int(min_samples)
            else:
                key = (
                    "n_clusters" if self.algorithm_choice in ["kmeans", "agglomerative"] 
                    else "n_components" if self.algorithm_choice == "gmm" 
                    else "bandwidth"
                )
                params[key] = float(response) if "." in response else int(response)
            
            self.cluster_model.set_params(**params)

            self.fit_model()
            self.evaluate_clusters()

        except Exception as e:
            print(f"Error parsing parameters: {e}. Using default parameters.")
            self.llm_error_flag = True  # Mark as error
            self.parameters_error_flag = True  # Mark as error


    # Action 3
    def choose_norm(self):
        """Choose the normalization method based on the data type of the columns."""

        self.reset_flags()

        column_types = {
            "categorical": [col for col in self.__backup_df.columns if self.__backup_df[col].dtype == "object"],
            "integer": [col for col in self.__backup_df.columns if self.__backup_df[col].dtype in ["int64", "int32"]],
            "boolean": [col for col in self.__backup_df.columns if self.__backup_df[col].dtype == "bool"],
            "float": [col for col in self.__backup_df.columns if self.__backup_df[col].dtype in ["float64", "float32"]]
        }

        prompt = f"""You need to normalize the following DataFrame: 
        {self.df}.
        The columns are grouped by type as follows:
        - Categorical: {column_types['categorical']}
        - Integer: {column_types['integer']}
        - Boolean: {column_types['boolean']}
        - Float: {column_types['float']}
        = Choose a normalization method for each type of data.
        = The options are:
        - Categorical: None, OneHotEncoding, LabelEncoding
        - Integer: StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
        - Boolean: None, BinaryScaler (scales to 0 and 1)
        - Float: StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
        Return ONLY the answer in the following JSON format:
        {{
        "categorical": "None",
        "integer": "StandardScaler",
        "boolean": "BinaryScaler",
        "float": "MinMaxScaler"
        }}
        """

        self.add_to_history({"role": "user", "content": prompt})
        response = self.generate(prompt)
        self.add_to_history({"role": "assistant", "content": response})

        print("Response:", response)

        norm_map = {
            "StandardScaler": StandardScaler(),
            "MinMaxScaler": MinMaxScaler(),
            "MaxAbsScaler": MaxAbsScaler(),
            "RobustScaler": RobustScaler(),
            "BinaryScaler": MinMaxScaler(feature_range=(0, 1)),  
            "OneHotEncoding": None,  
            "LabelEncoding": None,  
            "None": None  
        }

        try:
            # Interpret the response as a dictionary
            chosen_norms = eval(response)
            if not isinstance(chosen_norms, dict):
                raise ValueError("Invalid response format. Expected a dictionary.")

            temp_df = self.df.copy()

            # Normalization for each column 
            for dtype, norm_name in chosen_norms.items():
                if dtype not in column_types or norm_name not in norm_map:
                    raise ValueError(f"Invalid normalization type or method: {dtype}, {norm_name}")

                scaler = norm_map[norm_name]
                for col in column_types[dtype]:
                    if scaler is not None:
                        temp_df[col] = scaler.fit_transform(temp_df[[col]])
                    elif norm_name == "OneHotEncoding":
                        temp_df = pd.get_dummies(temp_df, columns=[col])
                    elif norm_name == "LabelEncoding":
                        le = LabelEncoder()
                        temp_df[col] = le.fit_transform(temp_df[col])

            self.df = temp_df
            self.data = self.df.values

            self.evaluate_clusters()

        except Exception as e:
            self.llm_error_flag = True  # Mark as error


    # Action 4
    def reset_data(self):
        """Reset the data to the original state."""
        self.data = self.__backup_data.copy()
        self.df = self.__backup_df.copy()      

        self.fit_model()  


    def reset_flags(self):
        self.llm_error_flag = False  
        self.parameters_error_flag = False 
        self.n_cluster_invalid_flag = False 



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
            self.n_cluster_invalid_flag = True

        else:
            self.evaluation_results = {
                "silhouette_score": silhouette_score(self.data, labels),
                "davies_bouldin_score": davies_bouldin_score(self.data, labels),
                "n_clusters": n_clusters}
        