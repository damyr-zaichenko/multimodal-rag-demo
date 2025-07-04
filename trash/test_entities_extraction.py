import spacy
from typing import List
import subprocess

def load_spacy_model(model_name="en_core_web_trf"):
    try:
        return spacy.load(model_name)
    except OSError:
        subprocess.run(["python", "-m", "spacy", "download", model_name], check=True)
        return spacy.load(model_name)
    

def extract_main_entities(query: str, model_name="en_core_web_trf") -> List[str]:
    try:
        nlp = spacy.load(model_name)
    except OSError:
        subprocess.run(["python", "-m", "spacy", "download", model_name], check=True)
        nlp = spacy.load(model_name)
    
    doc = nlp(query)
    entities = [ent.text for ent in doc.ents if ent.label_ in {"ORG", "PERSON", "PRODUCT"}]

    return entities
    

test_queries = [
    "What are Alibaba's latest open-source AI models?",
    "Tell me about OpenAI and Anthropic's new LLM releases.",
    "Is Google collaborating with DeepMind on Gemini 2?",
    "Who founded NVIDIA and what are their recent models?",
    "Does Amazon use GPT models in Alexa?",
    "Latest updates on Meta and Microsoft AI partnerships.",
    "Are there any Claude 3.5 updates from Anthropic?",
    "What are the best models released by Mistral AI?",
    "How does Tesla use AI in self-driving?",
    "Compare ChatGPT, Claude, and Gemini performance in coding.",
    "What companies are building healthcare AI tools?",
    "Are there any new AI tools for education?",
    "Which models are best for finance applications?",
    "Show me recent advancements from Samsung Research.",
    "What is Elon Musk's AI company working on?",
    "Did Apple release any AI features in iOS 18?",
    "Give me a list of models with open weights.",
    "Is Salesforce investing in AI research?",
    "What does IBM Watson do today?",
    "Are there any AI startups in Europe?"
]

if __name__ == '__main__':
    for query in test_queries:
        print(f"Query: {query}")
        print("Entities:", extract_main_entities(query))
        print("-" * 60)