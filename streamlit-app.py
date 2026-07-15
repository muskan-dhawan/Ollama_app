import streamlit as st
import os
import random
from sentence_transformers import SentenceTransformer
from groq import Groq

st.set_page_config(page_title="Particle Physics RAG Assistant", page_icon="⚛️", layout="wide")

# Model Configurations
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
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
    st.caption("Engine: Groq + Sentence Transformers")
    st.caption(f"Embeddings: `{EMBEDDING_MODEL}`")
    st.caption(f"LLM: `{LANGUAGE_MODEL}`")

# Main Page
st.title("⚛️ Particle Physics RAG Assistant")
st.write("Ask any question about particle physics! This app retrieves relevant facts from a local vector database and uses a fast cloud LLM via Groq to generate an answer.")

# Check for API Key
if "GROQ_API_KEY" in st.secrets:
    api_key = st.secrets["GROQ_API_KEY"]
else:
    api_key = os.environ.get("GROQ_API_KEY")

if not api_key:
    st.error("Missing GROQ_API_KEY. Please set it in Streamlit secrets or as an environment variable.")
    st.stop()

# Initialize Groq client
client = Groq(api_key=api_key)

# --- 1. Vector DB Setup & Caching ---
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)

@st.cache_resource
def initialize_vector_db():
    """Loads the dataset and computes embeddings once, caching the results."""
    
    # Check if file exists to prevent crashing
    if not os.path.exists(DATASET_PATH):
        # Creating a fallback file for demonstration if it doesn't exist
        with open(DATASET_PATH, 'w', encoding="utf-8") as f:
            f.write("1. The Standard Model of particle physics classifies all known elementary particles.\n")
            f.write("2. Quarks are elementary particles that combine to form hadrons, such as protons and neutrons.\n")
            f.write("3. There are six flavors of quarks: up, down, charm, strange, top, and bottom.\n")
            f.write("4. Leptons include electrons, muons, tau particles, and their associated neutrinos.\n")
            f.write("5. The Higgs boson is responsible for giving mass to other fundamental particles.\n")

    with open(DATASET_PATH, 'r', encoding="utf-8") as file:
        dataset = file.readlines()
    
    vector_db = []
    
    # Progress bar for visual feedback during startup embedding generation
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    embedder = load_embedding_model()
    
    for i, chunk in enumerate(dataset):
        chunk = chunk.strip()
        if not chunk:
            continue
        status_text.text(f"Embedding chunk {i+1}/{len(dataset)}...")
        try:
            embedding = embedder.encode(chunk).tolist()
            vector_db.append((chunk, embedding))
        except Exception as e:
            st.error(f"Error generating embeddings: {e}")
            return [], dataset
        progress_bar.progress((i + 1) / len(dataset))
        
    # Clear the loading indicators when done
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
    embedder = load_embedding_model()
    query_embedding = embedder.encode(query).tolist()
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
    # Filter out empty lines
    valid_facts = [fact.strip() for fact in raw_dataset if fact.strip()]
    if valid_facts:
        random_fact = random.choice(valid_facts)
        st.info(f"**Did you know?**\n\n{random_fact}", icon="💡")
    st.session_state.random_fact_requested = False


# --- 5. Chat Interface ---
st.divider()
st.subheader("Ask a question")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("e.g. What is the Higgs boson?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if not vector_db:
        assistant_msg = "⚠️ Vector DB is not loaded. Cannot retrieve context."
        st.session_state.messages.append({"role": "assistant", "content": assistant_msg})
        with st.chat_message("assistant"):
            st.markdown(assistant_msg)
    else:
        # Retrieve relevant context
        with st.spinner("Retrieving relevant facts..."):
            results = retrieve(prompt, vector_db, top_k=5)
            context = '\n'.join([f"- {chunk}" for chunk, score in results])

        # Show retrieved context in an expander
        with st.expander("📚 Retrieved Context", expanded=False):
            for chunk, score in results:
                st.markdown(f"- **[{score:.3f}]** {chunk}")

        # Build the prompt with context
        system_prompt = f"""You are a helpful assistant that answers questions about particle physics using ONLY the context below.
If the context does not contain the answer, say "I don't have enough information to answer that."

Context:
{context}"""

        # Generate response from LLM
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
