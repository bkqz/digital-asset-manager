import streamlit as st
import os
import time
from PIL import Image
from langchain_groq import ChatGroq
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

# --- Header & UX Guidance ---
st.title("🖼️ Multimodal Digital Asset Manager (RAG)")
st.markdown("Search your local image library using natural language and AI-generated captions.")

with st.expander("📖 How this system works", expanded=False):
    st.markdown("""
    This is a **Private Retrieval-Augmented Generation (RAG)** system. It does not search the public web.
    
    1.  **Step 1: Upload Assets** - Use the sidebar to upload local images (`.jpg`, `.png`) to your private Supabase bucket.
    2.  **Step 2: Indexing** - Click **'Process & Index'** to trigger Llama-4-Scout Vision analysis. This writes a detailed caption and generates a 768-dim vector for each image.
    3.  **Step 3: Semantic Search** - Describe a photo in plain English (e.g., 'a peaceful landscape') to find matches from your indexed collection.
    4.  **Step 4: Reasoning Chat** - Ask questions about the retrieved results (e.g., 'Which of these is most suitable for a nature blog?').
    """)

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

# --- Sidebar: Ingestion Control ---
with st.sidebar:
    st.header("📤 Upload Assets")
    uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    st.header("⚙️ Ingestion Engine")
    if st.button("Process & Index Assets"):
        if uploaded_files:
            with st.status("🚀 Processing Assets...", expanded=True) as status:
                for i, uploaded_file in enumerate(uploaded_files):
                    # Save locally temporarily
                    file_path = os.path.join(config.UPLOAD_DIR, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    st.write(f"Analyzing `{uploaded_file.name}` with Llama 4 Vision...")
                    
                    try:
                        ingest_image_to_pinecone(file_path)
                        st.write(f"✅ Indexed: `{uploaded_file.name}`")
                    except Exception as e:
                        st.error(f"❌ Error processing {uploaded_file.name}: {e}")
                        continue
                
                status.update(label="Index Update Complete!", state="complete", expanded=False)
            
            st.success(f"Indexed {len(uploaded_files)} assets in Pinecone & Supabase.")
            time.sleep(1)
            st.rerun()
        else:
            st.warning("Please select files first.")

    st.divider()
    st.info("Backend: Pinecone (768-dim) | HF Inference | Llama-4-Scout")

# --- Main UI: Semantic Search ---
st.subheader("🔍 Semantic Search")
query = st.text_input("Describe the image you're looking for:", placeholder="e.g. 'a photo of a seagull over the ocean' or 'urban architecture'")
st.caption("⚠️ The search is strictly limited to assets currently indexed in your private Supabase library.")

if query:
    with st.spinner("Retrieving from Pinecone..."):
        results = retrieve_similar_images(query, top_k=3)
        
        if results:
            st.subheader("🎯 Top Matches")
            st.info("💡 Note: Matches with a similarity score below **40%** may be less relevant to your query.")
            cols = st.columns(3)
            
            context_text = "Retrieved Image Context:\n"
            
            for i, res in enumerate(results):
                with cols[i]:
                    # Display from Supabase Public URL
                    st.image(res['file_path'], width='stretch')
                    st.metric("Similarity Match", f"{res['score']:.1%}")
                    with st.expander("📝 AI Analysis"):
                        st.write(res['caption'])
                
                context_text += f"- Image {i+1}: {res['caption']}\n"
            
            st.session_state.retrieved_context = context_text
        else:
            st.error("No relevant images found in your collection.")

# --- Chat Interface (Asset Reasoning) ---
st.divider()
st.subheader("💬 Asset Reasoning")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if chat_input := st.chat_input("Ask a question about the retrieved images..."):
    st.session_state.messages.append({"role": "user", "content": chat_input})
    with st.chat_message("user"):
        st.markdown(chat_input)

    with st.chat_message("assistant"):
        full_system_prompt = f"""
        You are an expert digital asset assistant.
        
        {st.session_state.retrieved_context}
        
        Rules:
        1. Answer based ONLY on the provided context of retrieved images.
        2. If no images are retrieved, politely ask the user to perform a search first.
        3. Be professional and concise.
        """
        
        messages = [
            SystemMessage(content=full_system_prompt),
            HumanMessage(content=chat_input)
        ]
        
        response = llm.invoke(messages)
        st.markdown(response.content)
        st.session_state.messages.append({"role": "assistant", "content": response.content})

# --- Footer ---
st.caption("Architecture: Multi-modal RAG-driven Digital Asset Manager | Powered by Groq, Pinecone, and Supabase.")
