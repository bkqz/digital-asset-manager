import streamlit as st
import os
import time
from PIL import Image
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
import src.config as config
from src.ingestion import ingest_image_to_pinecone
from src.retrieval import retrieve_similar_images

# --- Page Configuration ---
st.set_page_config(
    page_title="Multimodal Asset Manager",
    page_icon="🖼️",
    layout="wide"
)

st.title("🖼️ Multimodal Digital Asset Manager (RAG)")
st.markdown("Search your local image library using natural language and AI-generated captions.")

# --- Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "retrieved_context" not in st.session_state:
    st.session_state.retrieved_context = ""

# Initialize Reasoning LLM
llm = ChatGroq(
    model=config.REASONING_LLM_NAME,
    groq_api_key=config.GROQ_API_KEY,
    temperature=0.5
)

# --- Sidebar: Upload & Stats ---
with st.sidebar:
    st.header("📤 Upload Assets")
    uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    if st.button("Process & Index"):
        if uploaded_files:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, uploaded_file in enumerate(uploaded_files):
                # Save locally
                file_path = os.path.join(config.UPLOAD_DIR, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Ingest to Pinecone
                status_text.text(f"Indexing: {uploaded_file.name}...")
                try:
                    ingest_image_to_pinecone(file_path)
                except Exception as e:
                    st.error(f"❌ Error processing {uploaded_file.name}: {e}")
                    # Continue with next file or stop? For now, we continue.
                    continue
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            st.success(f"Successfully indexed {len(uploaded_files)} images!")
            time.sleep(2)
            st.rerun()
        else:
            st.warning("Please select files first.")

    st.divider()
    st.info("System: Pinecone (768-dim) | HF Inference | Llama-4-Scout")

# --- Main UI: Search & Retrieval ---
query = st.text_input("🔍 Search your assets (e.g., 'a peaceful landscape' or 'bird in flight')")

if query:
    with st.spinner("Searching..."):
        # Make sure to set PYTHONPATH if needed when running locally
        results = retrieve_similar_images(query, top_k=3)
        
        if results:
            st.subheader("🎯 Top Results")
            cols = st.columns(3)
            
            context_text = "Here are the most relevant images found:\n"
            
            for i, res in enumerate(results):
                with cols[i]:
                    # The file_path is now a Public URL from Supabase
                    st.image(res['file_path'], width='stretch', caption=f"Score: {res['score']:.3f}")
                    with st.expander("View Caption"):
                        st.write(res['caption'])
                
                context_text += f"- Image {i+1} (Path: {res['file_path']}): {res['caption']}\n"
            
            st.session_state.retrieved_context = context_text
        else:
            st.error("No relevant images found.")

# --- Chat Interface (The RAG Logic) ---
st.divider()
st.subheader("💬 Chat about your Assets")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if chat_input := st.chat_input("Ask a question about the retrieved images..."):
    # Add user message to state
    st.session_state.messages.append({"role": "user", "content": chat_input})
    with st.chat_message("user"):
        st.markdown(chat_input)

    # Generate Response
    with st.chat_message("assistant"):
        # Build prompt using chat context
        history = ""
        for msg in st.session_state.messages[-5:]: # Last 5 messages for history
            history += f"{msg['role']}: {msg['content']}\n"
            
        full_system_prompt = f"""
        You are a helpful assistant for a Digital Asset Manager.
        
        Context from retrieved images:
        {st.session_state.retrieved_context}
        
        Answer the user's question based ONLY on the provided context. If no images have been retrieved yet, ask the user to perform a search first. Be concise and professional.
        """
        
        messages = [
            SystemMessage(content=full_system_prompt),
            HumanMessage(content=chat_input)
        ]
        
        response = llm.invoke(messages)
        st.markdown(response.content)
        st.session_state.messages.append({"role": "assistant", "content": response.content})

# --- Footer ---
st.caption("Powered by Groq, Pinecone, and Hugging Face.")
