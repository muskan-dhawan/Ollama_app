import streamlit as st
import ollama
import os

st.set_page_config(page_title="Cricket RAG Assistant", layout="centered", page_icon="🏏")
st.title("🏏 Cricket Knowledge Assistant")
st.markdown("""
Welcome to the **Cricket Knowledge Assistant**! 
This app uses a local LLM via **Ollama** to answer your questions based on a specific knowledge base about cricket.
""")

# Model Configurations
EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'

# --- 1. Vector DB Setup & Caching ---
@st.cache_resource
def initialize_vector_db():
    """Loads the dataset and computes embeddings once, caching the results."""
    dataset_path = 'cricket-facts.txt'
    
    # Check if file exists to prevent crashing
    if not os.path.exists(dataset_path):
        # Creating a fallback file for demonstration if it doesn't exist
        with open(dataset_path, 'w') as f:
            f.write("Cricket is believed to have been invented in South East England in the 16th century.\n")
            f.write("A cricket pitch is exactly 22 yards (20.12 meters) long between the wickets.\n")
            f.write("Sachin Tendulkar holds the record for the most runs in international cricket.\n")

    with open(dataset_path, 'r',encoding="utf-8") as file:
        dataset = file.readlines()
    
    vector_db = []
    
    # Progress bar for visual feedback during startup embedding generation
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    for i, chunk in enumerate(dataset):
        chunk = chunk.strip()
        if not chunk:
            continue
        status_text.text(f"Embedding chunk {i+1}/{len(dataset)}...")
        try:
            embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]
            vector_db.append((chunk, embedding))
        except Exception as e:
            st.error(f"Error connecting to Ollama: {e}")
            return []
        progress_bar.progress((i + 1) / len(dataset))
        
    # Clear the loading indicators when done
    status_text.empty()
    progress_bar.empty()
    return vector_db

# Initialize the database
with st.spinner("Initializing Vector Database and generating embeddings..."):
    VECTOR_DB = initialize_vector_db()


# --- 2. Helper Functions ---
def cosine_similarity(a, b):
    dot_product = sum([x * y for x, y in zip(a, b)])
    norm_a = sum([x ** 2 for x in a]) ** 0.5
    norm_b = sum([x ** 2 for x in b]) ** 0.5
    if int(norm_a * norm_b) == 0:
        return 0
    return dot_product / (norm_a * norm_b)

## Retrieval function that takes a user query, computes its embedding, and finds the most similar chunks in the VECTOR_DB based on cosine similarity.
def retrieve(query, top_n=3):
  query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
  # temporary list to store (chunk, similarity) pairs
  similarities = []
  for chunk, embedding in VECTOR_DB:
    similarity = cosine_similarity(query_embedding, embedding)
    similarities.append((chunk, similarity))
  # sort by similarity in descending order, because higher similarity means more relevant chunks
  similarities.sort(key=lambda x: x[1], reverse=True)
  # finally, return the top N most relevant chunks
  return similarities[:top_n]


# --- 3. UI and Interaction ---

# Sidebar to show the retrieved context chunks behind the scenes
with st.sidebar:
    st.header("Database Info")
    st.success(f"Loaded {len(VECTOR_DB)} items from dataset.")
    st.markdown("---")
    st.subheader("Retrieved Context Window")
    context_placeholder = st.empty()
    context_placeholder.info("Ask a question to see the matching context chunks here!")


# User Input
st.markdown("### 🏏 Ask Away!")
input_query = st.text_input("Ask me a question about cricket:", placeholder="e.g., What is the length of a cricket pitch?")

if input_query:
    # Perform Retrieval
    retrieved_knowledge = retrieve(input_query)
    
    # Update the sidebar dynamically to display retrieved chunks and their scores
    with context_placeholder.container():
        for chunk, similarity in retrieved_knowledge:
            st.markdown(f"**Score:** `{similarity:.2f}`\n\n*{chunk}*")
            st.markdown("---")

    # Construct the instruction prompt
    instruction_prompt = f"""You are a helpful cricket expert chatbot.
Use ONLY the following pieces of context to answer the question. Do not use outside knowledge. If the answer is not in the context, say you don't know based on the provided facts.

Context:
{'\n'.join([f' - {chunk}' for chunk, similarity in retrieved_knowledge])}
"""

    st.markdown("### 🤖 Response:")
    
    # Dynamic placeholder for streaming the response
    response_placeholder = st.empty()
    full_response = ""
    
    try:
        stream = ollama.chat(
            model=LANGUAGE_MODEL,
            messages=[
                {'role': 'system', 'content': instruction_prompt},
                {'role': 'user', 'content': input_query},
            ],
            stream=True,
        )
        
        # Iterate over stream chunks and dynamically update the UI
        for chunk in stream:
            token = chunk['message']['content']
            full_response += token
            response_placeholder.markdown(full_response + "▌")
            
        # Final update to remove the cursor icon
        response_placeholder.markdown(full_response)
        
    except Exception as e:
        st.error(f"Error generating chat response: {e}")