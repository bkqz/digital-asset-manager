import os
import uuid
import time
from pinecone import Pinecone
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from huggingface_hub import InferenceClient
from supabase import create_client, Client
import src.config as config
from src.utils import encode_image, get_file_name

# Initialize Pinecone
pc = Pinecone(api_key=config.PINECONE_API_KEY)
index = pc.Index(host=config.PINECONE_HOST)

# Initialize Vision Model (via Groq)
vision_llm = ChatGroq(
    model=config.VISION_MODEL_NAME,
    groq_api_key=config.GROQ_API_KEY
)

# Initialize HF Inference Client for Embeddings
hf_client = InferenceClient(token=config.HF_TOKEN)

# Initialize Supabase
supabase: Client = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)

import mimetypes

def upload_to_supabase(image_path: str) -> str:
    """
    Uploads an image to Supabase Storage and returns the Public URL.
    """
    # The policy requires the file to be in the 'public/' folder
    file_name = f"public/{get_file_name(image_path)}"
    
    # Get mime type (e.g., image/jpeg or image/png)
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type:
        mime_type = "application/octet-stream"
    
    with open(image_path, "rb") as f:
        # Use upsert=True to allow overwriting if needed during testing
        res = supabase.storage.from_(config.SUPABASE_BUCKET_NAME).upload(
            path=file_name,
            file=f.read(),
            file_options={"content-type": mime_type, "upsert": "true"}
        )
        
        # Check for error in response (SDK can return a dict or object)
        if isinstance(res, dict) and res.get("error"):
            raise Exception(f"Upload Error: {res['error']}")
    
    # Get the public URL
    public_url_res = supabase.storage.from_(config.SUPABASE_BUCKET_NAME).get_public_url(file_name)
    return public_url_res

def get_image_caption(image_path: str) -> str:
    """
    Sends an image to the Vision LLM (Llama-3.2-Vision) to generate a detailed caption.
    """
    base64_image = encode_image(image_path)
    
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
    vector = hf_client.feature_extraction(
        text,
        model=config.EMBEDDING_MODEL_NAME
    )
    if isinstance(vector, list) and len(vector) > 0 and isinstance(vector[0], list):
        return vector[0]
    return vector

def ingest_image_to_pinecone(image_path: str):
    """
    Full pipeline: Image -> Supabase Upload -> Caption -> Embedding -> Pinecone Metadata (Public URL)
    """
    print(f"--- Ingesting: {get_file_name(image_path)} ---")
    
    # 1. Upload to Supabase
    print("Uploading to Supabase...")
    try:
        public_url = upload_to_supabase(image_path)
        # Check if we got a valid URL back (some SDK versions return dicts with errors)
        if not public_url or not str(public_url).startswith("http"):
            raise ValueError(f"Supabase upload failed or returned invalid URL: {public_url}")
        print(f"Public URL: {public_url}")
    except Exception as e:
        error_msg = f"Supabase Error: {e}"
        print(error_msg)
        raise Exception(error_msg) # Re-raise to be caught by app.py

    # 2. Generate Caption (The Vision Step)
    caption = get_image_caption(image_path)
    print(f"Generated Caption: {caption[:100]}...")
    
    # 3. Generate Embedding (HF Inference API)
    try:
        vector = get_huggingface_embeddings(caption)
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None, None

    # 4. Create Metadata
    # Now storing the public_url instead of the local path
    metadata = {
        "file_path": public_url, 
        "caption": caption,
        "file_name": get_file_name(image_path),
        "timestamp": time.time()
    }
    
    # 5. Upsert to Pinecone
    entry_id = str(uuid.uuid4())
    index.upsert(vectors=[(entry_id, vector, metadata)])
    
    print(f"Successfully ingested to Pinecone with ID: {entry_id}")
    return entry_id, caption

if __name__ == "__main__":
    import glob
    test_images = glob.glob(os.path.join(config.UPLOAD_DIR, "*.jpg"))
    if test_images:
        print(f"Found {len(test_images)} images. Ingesting first 3 for testing...")
        for img_path in test_images[:3]:
            ingest_image_to_pinecone(img_path)
    else:
        print(f"No images found in {config.UPLOAD_DIR}.")
