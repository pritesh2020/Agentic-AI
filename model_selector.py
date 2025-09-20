from langchain_ollama import OllamaEmbeddings, OllamaLLM, ChatOllama
from langchain_openai import ChatOpenAI
import os


OLLAMA_HOST = "http://127.0.0.1:11434"  # changed to local Ollama
EMBED_MODEL = "embeddinggemma"   # selected embedding model
LLM_MODEL = "llama3.1:8b"     # using the requested model

def get_ollama_models(temperature=0.7):
    llm = ChatOllama(
    model=LLM_MODEL,
    temperature=0.7,
    base_url=OLLAMA_HOST,
    verbose=True
    )
    return llm

def get_openai_models(temperature=0.7):
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    return llm

def get_models(platform: str, temperature=0.7):
    if platform == "ollama":
        return get_ollama_models(temperature=0.7)
    elif platform == "openai":
        return get_openai_models(temperature=0.7)
    else:
        raise ValueError("Unsupported platform. Choose 'ollama' or 'openai'.")
    



