import ollama


class LLMAgent:
    def __init__(self, model="llama3.1"):
        self.model = model
        # Conversation history memory
        self.history = []  

    def add_to_history(self, content):
        self.history.append(content)

        # Limits history to the last 10
        if len(self.history) > 10:
            self.history.pop(0)

    def generate(self, prompt):
        # Adds the prompt to the message history
        self.add_to_history({"role": "user", "content": prompt})
        response = ollama.chat(model=self.model, messages=self.history)

        # Adds model response to history
        self.add_to_history({"role": "assistant", "content": response["message"]["content"]})
        
        return response["message"]["content"]
    
"""
agent = LLMAgent()
response1 = agent.generate("Escreva uma função em Python (chamada fatorial()) para calcular o fatorial de um número inteiro. Não escreva nada além do código. Apenas forneça a função.")

print("Código gerado pelo modelo:\n")
print(response1)

try:
    # Código da resposta gerada 
    exec(response1)  
    
    # Teste a função gerada
    print("\nResultado da função de fatorial (fatorial(5)):", fatorial(5))  

except Exception as e:
    print(f"Erro ao executar o código gerado: {e}")
"""