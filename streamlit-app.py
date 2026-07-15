import streamlit as st
import os
import random
import requests
from groq import Groq

st.set_page_config(page_title="Particle Physics RAG Assistant", page_icon="⚛️", layout="wide")

# Model Configurations
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
LANGUAGE_MODEL = 'llama3-8b-8192'
DATASET_PATH = 'physics-facts.txt'

# --- UI Layout: Sidebar ---
with st.sidebar:
    st.title("⚛️ Physics Explorer")
    st.write("Learn about the fundamental particles and forces of the universe!")
    st.divider()
    
    if st.button("🎲 Give me a Random Fact", use_container_width=True):
        st.session_state.random_fact_requested = True
    else:
        if "random_fact_requested" not in st.session_state:
            st.session_state.random_fact_requested = False

    st.divider()
    st.caption("Engine: Groq + Hugging Face API")
    st.caption(f"Embeddings: `{EMBEDDING_MODEL}`")
    st.caption(f"LLM: `{LANGUAGE_MODEL}`")

# Main Page
st.title("⚛️ Particle Physics RAG Assistant")
st.write("Ask any question about particle physics! This app retrieves relevant facts from a local vector database and uses a fast cloud LLM via Groq to generate an answer.")

# Check for API Keys
groq_api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY"))

if not groq_api_key:
    st.error("Missing GROQ_API_KEY. Please set it in Streamlit secrets.")
    st.stop()

# Initialize Groq client
client = Groq(api_key=groq_api_key)

# --- 1. Vector DB Setup & Caching ---
def get_embedding(text):
    """Uses Hugging Face Inference API to get embeddings (No heavy libraries needed!)"""
    api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{EMBEDDING_MODEL}"
    # We will use a public fallback token or no token if it's a free model, but providing one is better to avoid rate limits
    # The all-MiniLM-L6-v2 model is freely accessible on HF Inference API without a token for low volume
    headers = {}
    response = requests.post(api_url, headers=headers, json={"inputs": text})
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"HF API Error: {response.text}")

@st.cache_resource
def initialize_vector_db():
    """Loads the dataset and computes embeddings once, caching the results."""
    
    if not os.path.exists(DATASET_PATH):
        with open(DATASET_PATH, 'w', encoding="utf-8") as f:
            f.write("1. The Standard Model of particle physics classifies all known elementary particles.\n")
            f.write("2. Quarks are elementary particles that combine to form hadrons, such as protons and neutrons.\n")
            f.write("3. There are six flavors of quarks: up, down, charm, strange, top, and bottom.\n")
            f.write("4. Leptons include electrons, muons, tau particles, and their associated neutrinos.\n")
            f.write("5. The Higgs boson is responsible for giving mass to other fundamental particles.\n")

    with open(DATASET_PATH, 'r', encoding="utf-8") as file:
        dataset = file.readlines()
    
    vector_db = []
    
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    for i, chunk in enumerate(dataset):
        chunk = chunk.strip()
        if not chunk:
            continue
        status_text.text(f"Embedding chunk {i+1}/{len(dataset)} via HuggingFace API...")
        try:
            embedding = get_embedding(chunk)
            # If the API returns a list of lists, grab the first one
            if isinstance(embedding[0], list):
                embedding = embedding[0]
            vector_db.append((chunk, embedding))
        except Exception as e:
            st.error(f"Error generating embeddings: {e}")
            return [], dataset
        progress_bar.progress((i + 1) / len(dataset))
        
    status_text.empty()
    progress_bar.empty()
    return vector_db, dataset


# --- 2. Cosine Similarity ---
def cosine_similarity(a, b):
    dot_product = sum([x * y for x, y in zip(a, b)])
    norm_a = sum([x ** 2 for x in a]) ** 0.5
    norm_b = sum([x ** 2 for x in b]) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0
    return dot_product / (norm_a * norm_b)


# --- 3. Retriever ---
def retrieve(query, vector_db, top_k=5):
    """Finds the top_k most relevant chunks for a given query."""
    raw_emb = get_embedding(query)
    query_embedding = raw_emb[0] if isinstance(raw_emb[0], list) else raw_emb
    
    similarities = []
    for chunk, embedding in vector_db:
        similarity = cosine_similarity(query_embedding, embedding)
        similarities.append((chunk, similarity))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]


# --- 4. Initialize DB on first load ---
vector_db, raw_dataset = initialize_vector_db()

if vector_db:
    st.success(f"✅ Vector DB loaded with {len(vector_db)} physics facts.")
else:
    st.warning("⚠️ Vector DB is empty. Check your model and dataset.")

# --- Random Fact Handler ---
if st.session_state.random_fact_requested and raw_dataset:
    valid_facts = [fact.strip() for fact in raw_dataset if fact.strip()]
    if valid_facts:
        random_fact = random.choice(valid_facts)
        st.info(f"**Did you know?**\n\n{random_fact}", icon="💡")
    st.session_state.random_fact_requested = False


# --- 5. Chat Interface ---
st.divider()
st.subheader("Ask a question")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("e.g. What is the Higgs boson?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if not vector_db:
        assistant_msg = "⚠️ Vector DB is not loaded. Cannot retrieve context."
        st.session_state.messages.append({"role": "assistant", "content": assistant_msg})
        with st.chat_message("assistant"):
            st.markdown(assistant_msg)
    else:
        with st.spinner("Retrieving relevant facts..."):
            results = retrieve(prompt, vector_db, top_k=5)
            context = '\n'.join([f"- {chunk}" for chunk, score in results])

        with st.expander("📚 Retrieved Context", expanded=False):
            for chunk, score in results:
                st.markdown(f"- **[{score:.3f}]** {chunk}")

        system_prompt = f"""You are a helpful assistant that answers questions about particle physics using ONLY the context below.
If the context does not contain the answer, say "I don't have enough information to answer that."

Context:
{context}"""

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = client.chat.completions.create(
                        model=LANGUAGE_MODEL,
                        messages=[
                            {'role': 'system', 'content': system_prompt},
                            {'role': 'user', 'content': prompt},
                        ]
                    )
                    assistant_msg = response.choices[0].message.content
                except Exception as e:
                    assistant_msg = f"❌ Error generating response: {e}"
                
                st.markdown(assistant_msg)

        st.session_state.messages.append({"role": "assistant", "content": assistant_msg})
