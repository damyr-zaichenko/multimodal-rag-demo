from typing import List, Dict, Tuple
from openai import OpenAI
import streamlit as st
import os

from dotenv import load_dotenv
load_dotenv(override=True)

class AnswerGenerator:
    MODEL_PRICING = {
        "gpt-4":      {"input": 0.01,   "output": 0.03},
        "gpt-4-turbo":{"input": 0.01,   "output": 0.03},
        "gpt-4o":     {"input": 0.005,  "output": 0.015},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    }

    def __init__(self, model: str = "gpt-4o", temperature: float = 0.3, max_tokens: int = 512):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI()

    def build_prompt(self, query: str, results: List[Tuple[Dict, float]]) -> str:
        context = "\n\n".join(
            f"Source {i+1} ({meta['article_url']}):\n{meta['text']}" 
            for i, (meta, _) in enumerate(results)
        )

        return f"""
    Answer the following question using only the information from the sources below.

    💬 Question: {query}

    📚 Sources:
    {context}

    Write a concise, well-structured answer grounded in the facts from the sources.
    Include the source numbers in square brackets (e.g., [1], [2]) to support key facts.
    Mention key findings, authors (if relevant), and model names or datasets where applicable.
    Avoid vague phrasing like "one study showed" — be specific when possible.
    """.strip()

    def generate(self, query: str, results: List[Tuple[Dict, float]]) -> str:
        prompt = self.build_prompt(query, results)

        #self.show_cost_estimate(prompt, self.max_tokens)

        response = self.client.chat.completions.create(model=self.model,
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant that answers questions using retrieved documents."},
            {"role": "user", "content": prompt}
        ],
        temperature=self.temperature,
        max_tokens=self.max_tokens)

        return response.choices[0].message.content

    def show_cost_estimate(self, prompt: str, output_tokens: int):
        input_tokens = len(prompt.split())  # approx; for real count use tiktoken
        model_key = self.model.lower()
        pricing = self.MODEL_PRICING.get(model_key)

        if not pricing:
            st.warning(f"No pricing data for model `{self.model}`.")
            return

        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        total_cost = input_cost + output_cost

        st.markdown(f"""
        **💸 Estimated Cost**
        - Model: `{self.model}`
        - Input tokens: ~{input_tokens}
        - Output tokens (max): {output_tokens}
        - Total estimated cost (USD): **${total_cost:.4f}** 
        - Total estimated cost (UAH): **₴{total_cost*41.52:.4f}**

        """)

    def rewrite_query(self, query: str) -> str:
        """
        Use the LLM to improve/clarify the query for better retrieval.
        """
        prompt = f"""
            You are an AI assistant helping improve search queries for a document retrieval system.
            Given a user query, rewrite it to make it more specific, unambiguous, and aligned with technical or factual documents.

            Original query:
            "{query}"

            Rewritten query:
            """.strip()

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that rewrites vague user queries into clearer and more specific ones."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=64,
        )

        return response.choices[0].message.content.strip()