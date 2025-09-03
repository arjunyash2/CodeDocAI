from langchain_community.llms import Ollama

def get_llm():
    # Run: ollama pull codellama
    return Ollama(model="tinyllama")  # or "llama2"
