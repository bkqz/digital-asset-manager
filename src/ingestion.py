import os
import uuid
import time
from pinecone import Pinecone
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from huggingface_hub import InferenceClient
import src.config as config
from src.utils import encode_image, get_file_name

# Initialize Pinecone
pc = Pinecone(api_key=config.PINECONE_API_KEY)
# Using host directly for faster connections (Stable URL method)
index = pc.Index(host=config.PINECONE_HOST)

# Initialize Vision Model (via Groq)
vision_llm = ChatGroq(
    model=config.VISION_MODEL_NAME,
    groq_api_key=config.GROQ_API_KEY
)

# Initialize HF Inference Client for Embeddings
hf_client = InferenceClient(token=config.HF_TOKEN)

def get_image_caption(image_path: str) -> str:
    """
    Sends an image to the Vision LLM (Llama-3.2-Vision) to generate a detailed caption.
    """
    base64_image = encode_image(image_path)
    
    # Standard format for Groq multimodal messages
    message = HumanMessage(
        content=[
            {"type": "text", "text": "Provide a concise but detailed caption for this image for a search database. Describe objects, colors, and context."},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            },
        ]
    )
    
    response = vision_llm.invoke([message])
    return response.content

def get_huggingface_embeddings(text: str) -> list:
    """
    Generates embeddings using the Hugging Face Inference API.
    Model: sentence-transformers/all-mpnet-base-v2 (768 dimensions)
    """
    # Using feature_extraction to get vector representation
    vector = hf_client.feature_extraction(
        text,
        model=config.EMBEDDING_MODEL_NAME
    )
    # The response can be a list of lists or a single list depending on input
    # For a single string, it returns a 1D list or a 2D list with one element.
    if isinstance(vector, list) and len(vector) > 0 and isinstance(vector[0], list):
        return vector[0]
    return vector

def ingest_image_to_pinecone(image_path: str):
    """
    Full pipeline: Image -> Caption -> Embedding (HF API) -> Pinecone Metadata + Vector
    """
    print(f"--- Ingesting: {get_file_name(image_path)} ---")
    
    # 1. Generate Caption (The Vision Step)
    caption = get_image_caption(image_path)
    print(f"Generated Caption: {caption[:100]}...")
    
    # 2. Generate Embedding (HF Inference API)
    # Wait for the model to be loaded on HF if necessary
    try:
        vector = get_huggingface_embeddings(caption)
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None, None

    # 3. Create Metadata
    # GEMINI.md mandate: include 'file_path', 'caption', and 'timestamp'
    metadata = {
        "file_path": image_path,
        "caption": caption,
        "file_name": get_file_name(image_path),
        "timestamp": time.time()
    }
    
    # 4. Upsert to Pinecone
    entry_id = str(uuid.uuid4())
    index.upsert(vectors=[(entry_id, vector, metadata)])
    
    print(f"Successfully ingested to Pinecone with ID: {entry_id}")
    return entry_id, caption

if __name__ == "__main__":
    # Example usage for testing
    import glob
    test_images = glob.glob(os.path.join(config.UPLOAD_DIR, "*.jpg"))
    if test_images:
        print(f"Found {len(test_images)} images. Ingesting first 3 for testing...")
        for img_path in test_images[:3]:
            ingest_image_to_pinecone(img_path)
    else:
        print(f"No images found in {config.UPLOAD_DIR}.")
