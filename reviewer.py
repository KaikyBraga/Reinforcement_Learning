from coder import Coder
from llm import LLMAgent

class Reviewer(LLMAgent):
    """ 
    This class represents the Reviewer. Evaluates clustering quality 
    based on evaluation metrics, suggests hyperparameter adjustments.
    """

    def __init__(self, coder_instance, model="llama3.1"):
        super().__init__(model)  
        self.coder = coder_instance

        base_prompt = "You are a helpful assistant. Evaluate the clustering quality based on reward and the errors of Coder LLM."
        self.history = [{"role": "user", "content": base_prompt}]


    def evaluate_reward(self, reward, new_silhouette, previous_silhouette, new_davies_bouldin, previous_davies_bouldin, new_k, size_penalty, lambda_k, lambda_size):
        prompt = f"""
            The 'calculate_reward' function calculates the reward for a clustering agent. The reward is based on three main components:

            1. Quality Metrics:
            - Silhouette Score: Measures the separation between clusters. An improvement in the Silhouette Score contributes positively to the reward.
            - Davies-Bouldin Score: Measures the compactness and separation of clusters. A reduction in the Davies-Bouldin Score contributes positively to the reward.

            2. Penalties:
            - Number of Clusters (k): Penalizes if the number of clusters is too high, with the penalty being proportional to the number of clusters.
            - Cluster Size: Penalizes clusters that have fewer elements than the expected minimum size (`t_min`).

            The formula for calculating the reward is as follows:

            reward = ((new_silhouette - previous_silhouette) - (new_davies_bouldin - previous_davies_bouldin)) - (lambda_k * new_k) - (lambda_size * size_penalty)

            Where:
            - new_silhouette ({new_silhouette:.3f}): Silhouette Score after the agent's action.
            - previous_silhouette ({previous_silhouette:.3f}): Silhouette Score before the agent's action.
            - new_davies_bouldin ({new_davies_bouldin:.3f}): Davies-Bouldin Score after the agent's action.
            - previous_davies_bouldin ({previous_davies_bouldin:.3f}): Davies-Bouldin Score before the agent's action.
            - new_k ({new_k}): Number of clusters after the action.
            - size_penalty ({size_penalty}): Penalty associated with clusters having fewer than `t_min` elements.
            - lambda_k ({lambda_k}): Penalty for an excessive number of clusters.
            - lambda_size ({lambda_size}): Penalty for small clusters.

            The goal of this function is to encourage the agent to improve the quality of the clusters while controlling the number and size of the clusters in a balanced way.

            Given a reward value calculated based on the 'calculate_reward' function, please provide a score from 0 to 10 for the result (equals to {reward:.3f}). RETURN ONLY THE NUMBER REFERRING TO THE SCORE, NOTHING ELSE.
            """
        
        response = self.generate(prompt)

        try:
            response = int(response.strip())
        except:
            response = self.evaluate_reward(reward, new_silhouette, previous_silhouette, new_davies_bouldin, previous_davies_bouldin, new_k, size_penalty, lambda_k, lambda_size)

        self.add_to_history({"role": "user", "content": prompt})
        self.add_to_history({"role": "assistant", "content": str(response)})

        self.coder.add_to_history({"role": "user", "content": f"The reward score for this clustering configuration is {response}/10."})


    def evaluate_delirious(self):
        if self.coder.llm_error_flag:
            response = "You didn't follow the instructions exactly. Please stick to the specifications more closely."
            self.coder.add_to_history({"role": "user", "content": response})


    def evaluate_parameters(self):
        if self.coder.parameters_error_flag:
            response = "Your parameters don't seem to be working for the data. Try adjusting the model and ensure the parameters are appropriate for the dataset."
            self.coder.add_to_history({"role": "user", "content": response})


    # def evaluate_accuracy(self):
    #     """Evaluate if clusters make sense based on an accuracy threshold (e.g., silhouette score > 0.5)"""
        
    #     silhouette_score, _ = self.coder.evaluate_clusters(metric="silhouette")
        
    #     if silhouette_score > 0.5:
    #         response = self.generate(f"The silhouette score is {silhouette_score}. The clusters are well separated.")
    #         return "Good cluster separation", response
    #     else:
    #         response = self.generate(f"The silhouette score is {silhouette_score}. The clusters need adjustments.")
    #         return "Consider adjusting parameters or algorithm", response
        

    # def check_efficiency(self):
    #     """Analyze model efficiency and complexity"""

    #     if isinstance(self.coder.cluster_model, KMeans):
    #         # Analyzing the number of clusters chosen
    #         message = f"KMeans with {self.coder.cluster_model.n_clusters} clusters may have high computational cost if n_clusters is large."
    #         response = self.generate(message)

    #         return message, response
        
    #     elif isinstance(self.coder.cluster_model, DBSCAN):
    #         message = f"DBSCAN with eps={self.coder.cluster_model.eps} and min_samples={self.coder.cluster_model.min_samples} requires efficient data scaling."
    #         response = self.generate(message)

    #         return message, response


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
