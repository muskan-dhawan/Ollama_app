# Cricket Knowledge Assistant (Ollama RAG Demo)

A Streamlit-based Retrieval-Augmented Generation (RAG) application that uses a local [Ollama](https://ollama.com/) instance to answer queries based on a provided knowledge base (`cricket-facts.txt`).

## Features
- **Local Embeddings**: Computes text embeddings locally using the `bge-base-en-v1.5-gguf` model via Ollama.
- **Local Generation**: Uses the `Llama-3.2-1B-Instruct-GGUF` model via Ollama for generating conversational responses.
- **Vector Search**: Calculates cosine similarity to retrieve the most relevant context chunks for a query.
- **Interactive UI**: Built with Streamlit for a fast and responsive web interface.

## Requirements
- Python >= 3.10
- [Ollama](https://ollama.com/) installed and running locally
- Required Python packages: `streamlit`, `ollama`

Before running the app, ensure you have pulled the required models in Ollama:
```bash
ollama run hf.co/CompendiumLabs/bge-base-en-v1.5-gguf
ollama run hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF
```

## Running the App

You can start the Streamlit application with the following command:

```bash
streamlit run streamlit-app.py
```

## How It Works
1. Upon starting, the app loads `cricket-facts.txt` and computes embeddings for each chunk using Ollama.
2. When you submit a question, it generates an embedding for your query.
3. The app finds the top matching facts using cosine similarity.
4. The retrieved context is passed alongside your query to the Llama 3.2 model to generate an accurate, context-aware answer.

## Contributors
- muskan
