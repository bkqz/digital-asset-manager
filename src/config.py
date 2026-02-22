import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# --- API Keys ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

# --- Model Configurations ---
# Vision Model: For image captioning
VISION_MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"

# Reasoning LLM: For RAG response synthesis
REASONING_LLM_NAME = "llama-3.3-70b-versatile"

# Embeddings: Using all-mpnet-base-v2 (768-dim) via HF Inference API
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_DIMENSION = 768

# --- Pinecone Configuration ---
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "digital-asset-manager")
_raw_host = os.getenv("PINECONE_HOST")
# Pinecone SDK's Index(host=...) expects the host without https:// prefix
PINECONE_HOST = _raw_host.replace("https://", "") if _raw_host else None

# --- Storage Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")

# Ensure directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)

def validate_config():
    """Validates that essential configuration and environment variables are set."""
    missing = []
    if not PINECONE_API_KEY:
        missing.append("PINECONE_API_KEY")
    if not PINECONE_HOST:
        missing.append("PINECONE_HOST")
    if not GROQ_API_KEY:
        missing.append("GROQ_API_KEY")
    if not HF_TOKEN:
        missing.append("HF_TOKEN")
    
    if missing:
        raise EnvironmentError(f"Missing environment variables: {', '.join(missing)}. "
                                "Please check your .env file.")
    print("Configuration validated successfully.")

if __name__ == "__main__":
    # Test loading and directory creation
    try:
        validate_config()
        print(f"Upload directory: {UPLOAD_DIR}")
    except EnvironmentError as e:
        print(f"Config validation error: {e}")
