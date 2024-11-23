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

    def __init__(self, df, data, evaluation_results_initial, model="llama3.1"):
        super().__init__(model)
        base_prompt = "You are a data scientist specialized in machine learning. Your task is to solve clustering problem. Return only the requested requirement, without additional explanations. Focus solely on the specifications provided in the prompt."
        self.history = [{"role": "user", "content": base_prompt}]
        self.df = df # Pandas DataFrame
        self.data = data # Array of df
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


    # Action 3
    def remove_outliers(self):
    
        columns = self.df.columns

        prompt = f"""
        You are evaluating a clustering task with the following metrics:
        - Silhouette Score: {self.evaluation_results['silhouette_score']}
        - Davies-Bouldin Index: {self.evaluation_results['davies_bouldin_score']}

        Suggest a percentage of the data that should remain after removing outliers. 
        This percentage should optimize the clustering metrics, ensuring a balance between compact clusters 
        (improving the Silhouette score) and minimizing overlap (reducing the Davies-Bouldin Index). 
        The percentage should represent the data that stays after outlier removal.

        RETURN A SINGLE NUMBER (percentage) BETWEEN 0 AND 1.
        """

        self.llm_error_flag = False  # Reset flag

        response = self.generate(prompt)

        print(response)
    
        try:
            pct = float(response)
            if not (0 < pct < 1):
                self.llm_error_flag = True
                return
            
        except Exception as e:
            self.llm_error_flag = False

        # Quantiles limits 
        q1 = (1 - pct) / 2
        q2 = pct + q1

        mask = pd.Series([True] * len(self.df))  
        
        for col in columns:
            Q1 = self.df[col].quantile(q1)  
            Q2 = self.df[col].quantile(q2)  

            # Atualization of the mask
            mask &= (self.df[col] >= Q1) & (self.df[col] <= Q2)

        self.df = self.df[mask].reset_index(drop=True)
        self.data = self.data[mask.values]


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
        prompt = f"""To perform good clustering, it is necessary for the data to be normalized.
                  Your task is to choose the method that best fits to normalize the following data array: {self.data}.
                  Choose from the following methods: MaxAbsScaler, MinMaxScaler, StandardScaler, RobustScaler, Normalizer.
                  RESPOND ONLY WITH THE NAME OF THE METHOD AS IT WAS GIVEN TO YOU!
                  """

        
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

# TESTE

np.random.seed(42)
data = np.random.randn(200, 5)  
df = pd.DataFrame(data, columns=[f"feature_{i}" for i in range(5)])


initial_metrics = {
    "silhouette_score": 0.3,  
    "davies_bouldin_score": 2.0,  
    "n_clusters": 3
}

coder = Coder(df=df, data=data, evaluation_results_initial=initial_metrics)


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
