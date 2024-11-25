import ollama

class LLMAgent:
    def __init__(self, model="llama3.1"):
        self.model = model
        # Conversation history memory
        self.history = []  

    def add_to_history(self, content):
        self.history.append(content)

        # Limits history to the last 14
        if len(self.history) > 14:
            self.history.pop(0)

    def generate(self, prompt):
        # Adds the prompt to the message history
        self.add_to_history({"role": "user", "content": prompt})
        response = ollama.chat(model=self.model, messages=self.history)

        # Adds model response to history
        self.add_to_history({"role": "assistant", "content": response["message"]["content"]})
        
        return response["message"]["content"]
