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

    def __init__(self, data, model="llama3.1", problem_description=""):
        super().__init__(model)
        self.base_prompt = "You are a data scientist specialized in machine learning. Your task is to write efficient and well-documented Python code to solve clustering problems using the scikit-learn library. Return only the requested code, without additional explanations. Focus solely on the specifications provided in the prompt and comment the code."
        self.history = [{"role": "user", "content": self.base_prompt}]
        self.problem_description = problem_description
        self.data = data
        # self.model = "llama3.1" 
        # Default algorithm
        self.algorithm_choice = "kmeans"  

    # Action 1    
    def choose_algorithm(self, **kwargs):
        """Select clustering algorithm: kmeans or dbscan, with specified kwargs parameters"""
        
        # Generate a prompt for the LLM about the choice of algorithm
        prompt = f"Write ONLY THE NAME of the algorithm kmeans or dbscan. Please, choose the new algorithm based on the provided kwargs parameters' {self.algorithm_choice}: {kwargs}."
        
        # Add the prompt to history before generating the response
        self.add_to_history({"role": "user", "content": prompt})

        response = self.generate(prompt)

        # Add the response to history after generating the response
        self.add_to_history({"role": "assistant", "content": response})

        print(response)

        if "kmeans" in self.algorithm_choice.lower():
            self.cluster_model = KMeans()
        elif "DBSCAN" in self.algorithm_choice.upper():
            self.cluster_model = DBSCAN()

    # Action 2
    def adjust_parameters(self, **kwargs):
        """Adjust the parameters for the selected clustering algorithm"""

        # Generate a prompt for the LLM about the choice of algorithm
        prompt = f"Given the selected clustering algorithm ({self.algorithm_choice}), return only the kwargs parameters as a string in the format 'param1=value1, param2=value2, ...'. The current kwargs to adjust are: {kwargs}."

        self.add_to_history({"role": "user", "content": prompt})

        response = self.generate(prompt)

        self.add_to_history({"role": "assistant", "content": response})

        print(response)

        # Convert the string returned by LLM to a dictionary format
        params = {}
        try:
            # Split the response into individual parameters
            for param in response.split(","):
                
                key, value = param.split("=")
                key = key.strip()
                value = value.strip()

                # Try to convert to correct type
                if value.isdigit():  
                    value = int(value)
                elif value.replace('.', '', 1).isdigit():  
                    value = float(value)

                params[key] = value
            
            print(f"Adjusted parameters: {params}")

            # Apply the parameters to the model
            self.cluster_model.set_params(**params)
            
        except Exception as e:
            print(f"Error adjusting parameters: {e}")
            
    
    def fit_model(self):
        """Fit the clustering model to the preprocessed data"""
        if self.cluster_model and self.data is not None:
            self.cluster_model.fit(self.data)
        else:
            raise ValueError("Model or data not initialized.")
        
        # Add the message to history about model fitting
        self.add_to_history({"role": "assistant", "content": "Clustering model has been fitted."})
        
        return self.cluster_model  


    def get_labels(self):
        """Get labels of the fitted model"""
        if hasattr(self.cluster_model, "labels_"):
            return self.cluster_model.labels_
        else:
            raise ValueError("Model has not been fitted yet.")
    
    
    def evaluate_clusters(self):
        """Evaluate clusters using multiple metrics and return them as a dictionary"""

        labels = self.get_labels()

        evaluation_results = {}

        silhouette = silhouette_score(self.data, labels)
        evaluation_results["silhouette"] = silhouette
        
        davies_bouldin = davies_bouldin_score(self.data, labels)
        evaluation_results["davies_bouldin"] = davies_bouldin
        
        return evaluation_results  


# Create some random data
data = np.random.rand(100, 2)  

# Create the Coder object
coder = Coder(data=data)

# Choose the algorithm
coder.choose_algorithm(previous_algorithm="kmeans", algorithm="kmeans", n_clusters=3)

# Adjust the parameters 
coder.adjust_parameters(n_clusters=5)

coder.fit_model()

print(coder.evaluate_clusters())