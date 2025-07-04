from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to("mps")  # or "cuda" if you have it

def rewrite_query_with_context(query: str) -> str:
    prompt = (
        f"Rewrite the following question to make it short, focused, and suitable for vector search in a RAG system.\n\n"
        f"Question: {query}\n"
        f"Rewritten:"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("mps")  # "cuda" or "cpu" if needed
    output = model.generate(**inputs, max_new_tokens=32, temperature=0.0, do_sample=False)
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    rewritten = decoded.split("Rewritten:")[-1].strip().split('\n')[0]
    return rewritten