import streamlit as st
import os
import random
import math
from collections import Counter
from groq import Groq

st.set_page_config(page_title="Particle Physics RAG Assistant", page_icon="⚛️", layout="wide")

# Model Configurations
LANGUAGE_MODEL = 'llama-3.1-8b-instant'
DATASET_PATH = 'particle_physics_facts.txt'

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
    st.caption("Engine: Groq + Local TF-IDF Search")
    st.caption("Embeddings: `Pure Python TF-IDF`")
    st.caption(f"LLM: `{LANGUAGE_MODEL}`")

# Main Page
st.title("⚛️ Particle Physics RAG Assistant")
st.write("Ask any question about particle physics! This app retrieves relevant facts from a local database and uses a fast cloud LLM via Groq to generate an answer.")

# Check for API Keys
groq_api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY"))

if not groq_api_key:
    st.error("Missing GROQ_API_KEY. Please set it in Streamlit secrets.")
    st.stop()

# Initialize Groq client
client = Groq(api_key=groq_api_key)

# --- 1. Dataset Setup ---
@st.cache_data
def load_dataset():
    if not os.path.exists(DATASET_PATH):
        return []
    with open(DATASET_PATH, 'r', encoding="utf-8") as file:
        dataset = [line.strip() for line in file.readlines() if line.strip()]
    return dataset

raw_dataset = load_dataset()

if raw_dataset:
    st.success(f"✅ Database loaded with {len(raw_dataset)} physics facts.")
else:
    st.warning("⚠️ Database is empty. Check your dataset.")

# --- 2. Pure Python TF-IDF Retriever ---
def retrieve(query, dataset, top_k=3):
    def stem(w):
        return w[:-1] if len(w) > 3 and w.endswith('s') and not w.endswith('ss') else w

    def tokenize(text):
        words = [w.strip('.,?!()[]{}"\'') for w in text.lower().split() if w.strip('.,?!()[]{}"\'')]
        return [stem(w) for w in words]
    
    tokenized_dataset = [tokenize(doc) for doc in dataset]
    query_tokens = tokenize(query)
    
    if not query_tokens:
        return []

    # Calculate IDF
    N = len(dataset)
    idf = {}
    for word in set(query_tokens):
        df = sum(1 for doc in tokenized_dataset if word in doc)
        idf[word] = math.log((N + 1) / (df + 1)) + 1
        
    # Calculate TF-IDF scores
    scores = []
    for i, doc in enumerate(tokenized_dataset):
        score = 0
        if not doc:
            continue
            
        doc_counts = Counter(doc)
        for word in query_tokens:
            if word in doc_counts:
                tf = doc_counts[word] / len(doc)
                score += tf * idf[word]
        scores.append((dataset[i], score))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]

# --- Random Fact Handler ---
if st.session_state.random_fact_requested and raw_dataset:
    random_fact = random.choice(raw_dataset)
    st.info(f"**Did you know?**\n\n{random_fact}", icon="💡")
    st.session_state.random_fact_requested = False


# --- 3. Chat Interface ---
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

    if not raw_dataset:
        assistant_msg = "⚠️ Database is not loaded. Cannot retrieve context."
        st.session_state.messages.append({"role": "assistant", "content": assistant_msg})
        with st.chat_message("assistant"):
            st.markdown(assistant_msg)
    else:
        with st.spinner("Retrieving relevant facts..."):
            results = retrieve(prompt, raw_dataset, top_k=3)
            # Only include results with a score > 0
            valid_results = [res for res in results if res[1] > 0]
            
            if not valid_results:
                context = "No relevant facts found for your query."
            else:
                context = '\n'.join([f"- {chunk}" for chunk, score in valid_results])

        with st.expander("📚 Retrieved Context", expanded=False):
            if not valid_results:
                st.markdown("No matching context found.")
            else:
                for chunk, score in valid_results:
                    st.markdown(f"- **[Score: {score:.3f}]** {chunk}")

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
