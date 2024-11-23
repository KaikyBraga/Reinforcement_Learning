from utils import calculate_reward, epsilon_greedy_decay, update_q_value  
import matplotlib.pyplot as plt

class System:
    def __init__(self, coder, reviewer):
        self.coder = coder
        self.reviewer = reviewer
        self.q_values_coder = {"Change": 0, "Fix": 0}  
        self.actions = ["Change", "Fix"]
        self.labels = None
        self.results = []
        self.results_flag = []


    def train(self, epochs, epsilon, epsilon_min, decay_rate, size_penalty, lambda_k, lambda_size, X):
        previous_labels = self.coder.get_labels()
        previous_silhouette = self.coder.evaluation_results["silhouette_score"]
        previous_davies_bouldin = self.coder.evaluation_results["davies_bouldin_score"]

        for epoch in range(epochs):
            # epsilon greedy
            action, epsilon = self.epsilon_greedy_decay(self.actions, self.q_values_coder, epsilon, epsilon_min, decay_rate)

            if action == "Change":
                self.coder.choose_algorithm()
            else:
                self.coder.adjust_parameters()

            new_labels = self.coder.get_labels()

            reward = calculate_reward(X, previous_labels, new_labels, lambda_k, lambda_size, t_min=5)

            new_silhouette = self.coder.evaluation_results["silhouette_score"]
            new_davies_bouldin = self.coder.evaluation_results["davies_bouldin_score"]
            new_k = self.coder.evaluation_results["n_clusters"]

            self.reviewer.evaluate_reward(reward, new_silhouette, previous_silhouette, new_davies_bouldin, previous_davies_bouldin, new_k, size_penalty, lambda_k, lambda_size)
            self.reviewer.evaluate_delirious()
            self.reviewer.evaluate_parameters()

            update_q_value(self.q_values_coder, action, reward)

            previous_labels = new_labels
            previous_silhouette = new_silhouette 
            previous_davies_bouldin = new_davies_bouldin

            self.results.append([self.coder.evaluation_results, reward, action])
            self.results_flag.append((self.coder.llm_error_flag, self.coder.parameters_error_flag))

        self.labels = self.coder.get_labels()
        return self.results, self.results_flag


    def plot_rewards(self):
        rewards = [result[1] for result in self.results]  

        plt.figure(figsize=(10, 6))
        plt.plot(range(len(rewards)), rewards, marker="o", color="b", label="Recompensa")
        plt.title("Evolução das Recompensas ao Longo do Treinamento")
        plt.xlabel("Épocas")
        plt.ylabel("Recompensa")
        plt.grid(True)
        plt.legend()
        plt.show()


    def plot_action_frequency(self):
        action_count = {"Change": 0, "Fix": 0}

        for result in self.results:
            action = result[2]  
            action_count[action] += 1

        actions = list(action_count.keys())
        frequencies = list(action_count.values())

        plt.figure(figsize=(8, 6))
        plt.bar(actions, frequencies, color=["#1f77b4", "#ff7f0e"])
        plt.title("Frequência das Ações ao Longo do Treinamento")
        plt.xlabel("Ações")
        plt.ylabel("Frequência")
        plt.show()


    def plot_error_flags(self):
        error_flags_count = {"LLM Error": 0, "Parameter Error": 0}

        for flag in self.results_flag:
            llm_error, param_error = flag
            if llm_error:
                error_flags_count["LLM Error"] += 1
            if param_error:
                error_flags_count["Parameter Error"] += 1

        flags = list(error_flags_count.keys())
        counts = list(error_flags_count.values())

        plt.figure(figsize=(8, 6))
        plt.bar(flags, counts, color=["#d62728", "#2ca02c"])
        plt.title("Frequência das Flags de Erro ao Longo do Treinamento")
        plt.xlabel("Tipo de Erro")
        plt.ylabel("Número de Ocorrências")
        plt.show()


    def plot_average_reward_per_action(self):
        rewards_by_action = {"Change": [], "Fix": []}

        for result in self.results:
            action = result[2]  
            reward = result[1]
            rewards_by_action[action].append(reward)

        average_rewards = {action: sum(rewards) / len(rewards) for action, rewards in rewards_by_action.items()}

        actions = list(average_rewards.keys())
        avg_rewards = list(average_rewards.values())

        plt.figure(figsize=(8, 6))
        plt.bar(actions, avg_rewards, color=["#1f77b4", "#ff7f0e"])
        plt.title("Recompensa Média por Ação")
        plt.xlabel("Ação")
        plt.ylabel("Recompensa Média")
        plt.show()


    def plot_reward_distribution(self):
        rewards = [result[1] for result in self.results]

        plt.figure(figsize=(10, 6))
        plt.hist(rewards, bins=30, color="blue", alpha=0.7)
        plt.title("Distribuição das Recompensas")
        plt.xlabel("Recompensa")
        plt.ylabel("Frequência")
        plt.grid(True)
        plt.show()


    def plot_all(self):
        self.plot_rewards()
        self.plot_action_frequency()
        self.plot_error_flags()
        self.plot_average_reward_per_action()
        self.plot_reward_distribution()

# class Environment:

#     def __init__(self, coder, reviewer):
#         self.coder = coder
#         self.reviewer = reviewer
#         self.state = {}
#         self.algorithms = ["kmeans", "dbscan"]

       
#     def execute(self):
#         self.coder.preprocess_data()
        
#         self.coder.choose_algorithm(algorithm=random.choice(self.algorithms), n_clusters=2) 
#         self.coder.fit_model()
        
#         self.state["labels"] = self.coder.get_labels()
       

#     def evaluate(self):
#         accuracy_eval = self.reviewer.evaluate_accuracy()
#         efficiency_eval = self.reviewer.check_efficiency()
#         return {"accuracy": accuracy_eval, "efficiency": efficiency_eval}

    

# class RewardSystem:
#     # TODO: Melhorar o sistema de recompensas.

#     def __init__(self):
#         self.reward = 0
    

#     def calculate_reward(self, evaluations):
#         if evaluations["accuracy"] == "Good cluster separation":
#             self.reward += 5
#         else:
#             self.reward -= 5
        
#         if "high computational cost" in evaluations["efficiency"]:
#             self.reward -= 2
#         else:
#             self.reward += 2
        
#         return self.reward
    

# class Trainer:
#     def __init__(self, coder, reviewer, environment, reward_system, epochs=100):
#         self.coder = coder
#         self.reviewer = reviewer
#         self.environment = environment
#         self.reward_system = reward_system
#         self.epochs = epochs
    

#     def train(self):
#         for epoch in range(self.epochs):
#             print(f"Epoch {epoch+1}/{self.epochs}")
#             self.environment.execute()
#             evaluations = self.environment.evaluate()
#             reward = self.reward_system.calculate_reward(evaluations)
#             print(f"Reward: {reward}")



# class DataLoader:
#     def __init__(self, data):
#         self.data = data


#     def split_data(self, test_size=0.25):
#         # TODO: Embaralhar os dados.

#         train_size = int(len(self.data) * (1 - test_size))
#         return self.data[:train_size], self.data[train_size:]
    

# class Logger:
#     def __init__(self):
#         self.logs = []
    

#     def log(self, epoch, evaluations, reward):
#         self.logs.append({"epoch": epoch, "evaluations": evaluations, "reward": reward})
    

#     def show_logs(self):
#         for log in self.logs:
#             print(log)
