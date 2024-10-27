import random


class Environment:

    def __init__(self, coder, reviewer):
        self.coder = coder
        self.reviewer = reviewer
        self.state = {}
        self.algorithms = ["kmeans", "dbscan"]
       
       
    def execute(self):
        self.coder.preprocess_data()
        
        self.coder.choose_algorithm(algorithm=random.choice(self.algorithms), n_clusters=2) 
        self.coder.fit_model()
        
        self.state["labels"] = self.coder.get_labels()
       

    def evaluate(self):
        accuracy_eval = self.reviewer.evaluate_accuracy()
        efficiency_eval = self.reviewer.check_efficiency()
        return {"accuracy": accuracy_eval, "efficiency": efficiency_eval}

    

class RewardSystem:
    # TODO: Melhorar o sistema de recompensas.

    def __init__(self):
        self.reward = 0
    

    def calculate_reward(self, evaluations):
        if evaluations["accuracy"] == "Good cluster separation":
            self.reward += 5
        else:
            self.reward -= 5
        
        if "high computational cost" in evaluations["efficiency"]:
            self.reward -= 2
        else:
            self.reward += 2
        
        return self.reward
    

class Trainer:
    def __init__(self, coder, reviewer, environment, reward_system, epochs=100):
        self.coder = coder
        self.reviewer = reviewer
        self.environment = environment
        self.reward_system = reward_system
        self.epochs = epochs
    

    def train(self):
        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}/{self.epochs}")
            self.environment.execute()
            evaluations = self.environment.evaluate()
            reward = self.reward_system.calculate_reward(evaluations)
            print(f"Reward: {reward}")



class DataLoader:
    def __init__(self, data):
        self.data = data


    def split_data(self, test_size=0.25):
        # TODO: Embaralhar os dados.

        train_size = int(len(self.data) * (1 - test_size))
        return self.data[:train_size], self.data[train_size:]
    

class Logger:
    def __init__(self):
        self.logs = []
    

    def log(self, epoch, evaluations, reward):
        self.logs.append({"epoch": epoch, "evaluations": evaluations, "reward": reward})
    

    def show_logs(self):
        for log in self.logs:
            print(log)
