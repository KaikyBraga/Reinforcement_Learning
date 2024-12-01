import random
from sklearn.metrics import silhouette_score, davies_bouldin_score
import numpy as np


def calculate_reward(X, previous_labels, new_labels, 
                     lambda_silhouette=0.5, lambda_davies=0.5, 
                     lambda_k=0.1, lambda_size=0.5, t_min=5):
    """
    Calculates the reward for a clustering agent with weighted metrics.

    Parameters:
    - X: array-like, shape (n_samples, n_features)
        Data used to calculate the metrics.
    - previous_labels: array-like, shape (n_samples,)
        Cluster labels before the action.
    - new_labels: array-like, shape (n_samples,)
        Cluster labels after the action.
    - lambda_silhouette: float
        Weight for the Silhouette Score in the reward calculation.
    - lambda_davies: float
        Weight for the Davies-Bouldin Index (inverted) in the reward calculation.
    - lambda_k: float
        Penalty for an excessive number of clusters.
    - lambda_size: float
        Penalty for clusters with few elements.
    - t_min: int
        Minimum expected cluster size.

    Returns:
    - reward: float
        Calculated reward.
    """

    # Calculate metrics
    previous_silhouette = silhouette_score(X, previous_labels) if len(set(previous_labels)) > 1 else -1
    new_silhouette = silhouette_score(X, new_labels) if len(set(new_labels)) > 1 else -1
    
    previous_davies_bouldin = davies_bouldin_score(X, previous_labels)
    new_davies_bouldin = davies_bouldin_score(X, new_labels)

    # Transform Davies-Bouldin to its inverse
    previous_davies_bouldin_inv = 1 / previous_davies_bouldin
    new_davies_bouldin_inv = 1 / new_davies_bouldin

    # Base reward (improvement in metrics)
    silhouette_improvement = new_silhouette - previous_silhouette
    davies_improvement = new_davies_bouldin_inv - previous_davies_bouldin_inv

    metric_reward = (lambda_silhouette * silhouette_improvement) + (lambda_davies * davies_improvement)

    # Penalty for the number of clusters
    new_k = len(set(new_labels))
    k_penalty = lambda_k * new_k

    # Penalty for small clusters
    cluster_sizes = [sum(new_labels == cluster) for cluster in set(new_labels)]
    size_penalty = lambda_size * sum(1 for size in cluster_sizes if size < t_min)

    # Final reward
    reward = metric_reward - k_penalty - size_penalty

    return reward


def epsilon_greedy_decay(actions, q_values, epsilon, epsilon_min, decay_rate):
    """
    Epsilon-greedy action selection with decay.
    
    Args:
        actions (list): List of possible actions (e.g., algorithms or metrics).
        q_values (dict): Dictionary mapping actions to their estimated Q-values.
        epsilon (float): Current probability of exploring a random action.
        epsilon_min (float): Minimum value for epsilon to ensure some exploration.
        decay_rate (float): Rate at which epsilon decays (e.g., 0.99 per step).
        step (int): Current step in the process (used to track decayed epsilon).
        
    Returns:
        str: Selected action based on epsilon-greedy policy.
        float: Updated epsilon value after decay.
    """
    # Decay epsilon with each step, ensuring it doesn't go below epsilon_min
    epsilon = max(epsilon_min, epsilon * decay_rate)
    
    # Epsilon-greedy selection
    if random.random() < epsilon:
        # Explore: Random action
        action = random.choice(actions)  
    else:
        # Exploit: Best action
        action = max(q_values, key=q_values.get)  
    
    return action, epsilon


def update_q_value(q_values, action, reward, alpha=0.1):
    """
    Update Q-value for a given action.
    
    Args:
        q_values (dict): Current Q-values for actions.
        action (str): Action whose Q-value is being updated.
        reward (float): Observed reward for the action.
        alpha (float): Learning rate (0 <= alpha <= 1).
        
    Returns:
        None
    """
    q_values[action] = q_values[action] + alpha * (reward - q_values[action])
