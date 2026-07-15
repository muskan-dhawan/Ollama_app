# ⚛️ Particle Physics RAG Assistant

**Live Demo:** [https://ollamaapp-z5gyxpi7wrubtmtjgtqbbp.streamlit.app/](https://ollamaapp-z5gyxpi7wrubtmtjgtqbbp.streamlit.app/)

## Overview
A lightweight, lightning-fast Retrieval-Augmented Generation (RAG) web application that answers questions about Particle Physics. The app uses a custom pure-Python TF-IDF search algorithm to retrieve facts from a local dataset and leverages **Meta's LLaMA 3.1 8B** model (via Groq API) to generate highly accurate and detailed responses.

## 🚀 Features
- **Retrieval-Augmented Generation (RAG):** Answers user questions by grounding responses in a local particle physics knowledge base.
- **Omniscient Fallback:** If the local database lacks the answer, the LLM utilizes its own vast pre-trained knowledge base to accurately explain any physics concept.
- **Custom NLP Search:** Built from scratch with a custom, dependency-free TF-IDF & Stemming text search algorithm, entirely eliminating memory crashes and API limits.
- **Blazing Fast AI:** Uses the Groq LPU API to stream LLaMA 3.1 inference at hundreds of tokens per second.
- **Clean UI:** Built with Streamlit for a highly responsive, modern chat interface.

## 🛠️ Technology Stack
- **Frontend:** Streamlit
- **LLM Engine:** Groq API (`llama-3.1-8b-instant`)
- **Embeddings/Search:** Custom Pure Python TF-IDF (Term Frequency-Inverse Document Frequency)
- **Language:** Python 3.9+

## ⚙️ Local Installation
If you'd like to run this project locally:

1. Clone the repository:
   ```bash
   git clone https://github.com/muskan-dhawan/Ollama_app.git
   cd Ollama_app
   ```
2. Install the lightweight requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your Streamlit Secrets (`.streamlit/secrets.toml`):
   ```toml
   GROQ_API_KEY="your_groq_api_key_here"
   ```
4. Run the app:
   ```bash
   streamlit run streamlit-app.py
   ```

## 👨‍💻 Contributor
- **Muskan Dhawan** ([@muskan-dhawan](https://github.com/muskan-dhawan))