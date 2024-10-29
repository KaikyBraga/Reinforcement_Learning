import ollama

class LLMAgent:
    def __init__(self, model="llama3.1"):
        self.model = model
        # Implemente alguma forma de memória para armazenar o histórico de conversas.

    def generate(self, prompt):
        messages = [
            {'role': 'user',
                'content': prompt,
             }
        ]
        response = ollama.chat(model=self.model)
        return response
