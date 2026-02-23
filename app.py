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
st.subheader("Talk to your image library")
st.markdown("""
    Find any asset in seconds using natural language. 
    This app automatically captions, indexes, and reasons over your private collection of images.
""")

with st.expander("🚀 Get Started in 4 Steps", expanded=False):
    st.markdown("""
    1. **Upload**: Drop your images into the sidebar.
    2. **Index**: Click 'Sync Library' to let Llama-4 analyze and vectorize your assets.
    3. **Search**: Type a description (e.g., *'sunset over the skyline'*) to find matches.
    4. **Chat**: Ask follow-up questions about your retrieved results.
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
    st.header("📥 Add New Assets")
    uploaded_files = st.file_uploader(
        "Drop files here", 
        type=["jpg", "jpeg", "png"], 
        accept_multiple_files=True,
        help="Supported formats: JPG, PNG. Max 5MB per file."
    )
    
    st.header("⚙️ Ingestion Engine")
    if st.button("Process & Index Assets"):
        if uploaded_files:
            with st.status("Analyzing visual data...", expanded=True) as status:
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
st.subheader("🔍 Find an Asset")
query = st.text_input("Search by description", placeholder="e.g. 'office space' or 'man looking at beautiful lanscape'", label_visibility="collapsed")
st.caption("⚠️ The search is strictly limited to assets currently indexed in Pinecone.")

if query:
    with st.spinner("Searching your library..."):
        results = retrieve_similar_images(query, top_k=3)
        
        if results:
            # Use st.container for a cleaner "Card" look
            with st.container(border=True):
                st.write(f"### 🎯 Results for: '*{query}*'")
                st.info("💡 Note: Matches with a confidence score below **40%** may be less relevant.")
                
                cols = st.columns(3)
                context_text = "Retrieved Image Context:\n"
                
                for i, res in enumerate(results):
                    with cols[i]:
                        # High-quality image display
                        st.image(res['file_path'], use_container_width=True)
                        
                        # Logic for color-coded metrics
                        score = res['score']
                        # 'normal' = green, 'off' = gray/red in some Streamlit themes
                        color_mode = "normal" if score > 0.4 else "off" 
                        
                        st.metric(
                            label="AI Confidence", 
                            value=f"{score:.0%}", 
                            delta_color=color_mode
                        )
                        
                        with st.expander("👁️ View AI Insights"):
                            st.caption("Generated Visual Caption:")
                            st.write(res['caption'])
                    
                    context_text += f"- Image {i+1}: {res['caption']}\n"
                
                # Store context for the Reasoning Chat
                st.session_state.retrieved_context = context_text
        else:
            st.error("No relevant images found. Try a different description or ensure your assets are indexed.")

# --- Chat Interface (Asset Reasoning) ---
st.divider()
st.subheader("💬 Ask the Assistant")
st.caption("Ask questions about the images found above. The assistant only knows what it sees in the search results.")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if chat_input := st.chat_input("e.g., 'Which of these photos is best for a social media header?'"):
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
st.divider()
st.caption("Architecture: Multi-modal RAG-driven Digital Asset Manager | Powered by Groq, Pinecone, and Supabase.")
