# Multimodal RAG System

## Overview

This project implements a  **Multimodal Retrieval-Augmented Generation (RAG)** system. It retrieves relevant news articles and associated images from [The Batch](https://www.deeplearning.ai/the-batch/), based on user queries.

The goal is to combine text and image retrieval into a simple, working prototype that helps users explore content more effectively. A lightweight UI is included for testing and showcasing results.

## Core Features

- **Scrapes Articles and Images**  
  Collects news content from The Batch, including visual media.

- **Embeds Text and Images**  
  Uses language and vision models (e.g., SentenceTransformers, CLIP) to generate embeddings.

- **Multimodal Vector Search**  
  Stores embeddings in FAISS and retrieves results based on similarity to the query.

- **Optional LLM Integration**  
  Uses GPT-4o or similar models to summarize or explain retrieved content.

- **Streamlit UI**  
  Provides a basic interface where users can input queries and see retrieved articles with images.

### 📺 [Watch the demo](https://youtu.be/m_wMoOWFKD0)

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/damyr-zaichenko/multimodal_rag_demo.git
cd multimodal_rag_demo
```

### 2. Set up the environment
```bash
python -m venv .venv
source .venv/bin/activate  # or use .venv\Scripts\activate on Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your OpenAI API key

Create a `.env` file:
```
OPENAI_API_KEY=your_openai_key_here
```

Or configure using `st.secrets` for Streamlit.

### 5. Launch the app
```bash
streamlit run main.py
```

## Directory Structure

```
multimodal-rag/
├── data/              # Scraped and processed content
├── models/            # Embedding and captioning logic
├── ui/                # Streamlit interface
├── utils/             # Scraping, cleaning, embedding helpers
├── vector_store/      # FAISS indexing and retrieval
├── main.py            # App entry point
├── requirements.txt
└── README.md
```

## Technologies Used

- Python 3.10+
- Streamlit
- BeautifulSoup / requests
- SentenceTransformers, CLIP
- FAISS for vector search
- OpenAI GPT models
- PIL / OpenCV

## Evaluation

Retrieval performance can be tested using:
- Recall@K, Precision@K
- Manual inspection via UI

## Deployment Options

This app can optionally be deployed to:
- Streamlit Cloud
- Hugging Face Spaces
- Docker + GCP/AWS (add Dockerfile if needed)
