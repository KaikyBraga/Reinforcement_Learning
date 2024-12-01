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


    def evaluate_reward(self, reward, new_silhouette, previous_silhouette, new_davies_bouldin, previous_davies_bouldin, new_k, size_penalty, lambda_k, lambda_size, lambda_silhouette, lambda_davies):
        
        prompt = f"""
            The 'calculate_reward' function evaluates the quality of clustering results and computes a reward based on multiple components. The goal is to balance the improvement in cluster quality with penalties for undesirable configurations. The reward calculation is as follows:

            ### Components of the Reward:
            
            **1. Quality Metrics**:
            - **Silhouette Score**: Measures cluster separation. An improvement (positive difference between `new_silhouette` and `previous_silhouette`) increases the reward, scaled by `lambda_silhouette`.
            - **Davies-Bouldin Score**: Measures cluster compactness and separation. A lower score is better, so the reward considers an *inverse improvement* (`1 / new_davies_bouldin - 1 / previous_davies_bouldin`), scaled by `lambda_davies`.

            **2. Penalty Metrics**:
            - **Number of Clusters (k)**: Penalizes if the number of clusters (`new_k`) is too high. The penalty is proportional to `lambda_k * new_k`.
            - **Cluster Size**: Penalizes configurations with small clusters (clusters smaller than a defined threshold `t_min`). The penalty scales with `lambda_size * size_penalty`, where `size_penalty` is the count of such small clusters.

            ### Reward Formula:
            The reward is calculated as:
            
            reward = ((lambda_silhouette * silhouette_improvement) + (lambda_davies * davies_improvement)) - (lambda_k * new_k) - (lambda_size * size_penalty)

            **Where**:
            - `silhouette_improvement = new_silhouette - previous_silhouette`
            - `davies_improvement = 1 / new_davies_bouldin - 1 / previous_davies_bouldin`
            - `new_silhouette` ({new_silhouette:.3f}): Silhouette Score after the agent's action.
            - `previous_silhouette` ({previous_silhouette:.3f}): Silhouette Score before the agent's action.
            - `new_davies_bouldin` ({new_davies_bouldin:.3f}): Davies-Bouldin Score after the agent's action.
            - `previous_davies_bouldin` ({previous_davies_bouldin:.3f}): Davies-Bouldin Score before the agent's action.
            - `new_k` ({new_k}): Number of clusters after the action.
            - `size_penalty` ({size_penalty}): Penalty for clusters smaller than `t_min`.
            - `lambda_k` ({lambda_k}): Weight for penalizing excessive clusters.
            - `lambda_size` ({lambda_size}): Weight for penalizing small clusters.
            - `lambda_silhouette` ({lambda_silhouette}): Weight for improvements in the Silhouette Score.
            - `lambda_davies` ({lambda_davies}): Weight for improvements in the Davies-Bouldin Score.

            ### Objective:
            This function is designed to incentivize the agent to maximize the quality of the clustering solution, as measured by Silhouette and Davies-Bouldin Scores, while keeping the number of clusters and the cluster sizes within reasonable bounds.

            Given the calculated reward ({reward:.3f}), please assign a score from 0 to 10 based on its effectiveness. RETURN ONLY THE NUMBER REFERRING TO THE SCORE, NOTHING ELSE.
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

    def evaluate_n_clusters(self):
        if self.coder.n_cluster_invalid_flag:
            response = "The number of clusters can't be less than 2! Adjust the model and ensure that the number of clusters are appropriate for the dataset."
            self.coder.add_to_history({"role": "user", "content": response})