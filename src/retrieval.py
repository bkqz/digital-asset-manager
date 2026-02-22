import sys
from pinecone import Pinecone
from huggingface_hub import InferenceClient
import src.config as config

# Initialize Pinecone
pc = Pinecone(api_key=config.PINECONE_API_KEY)
# Using the stable Host URL method as configured in config.py
index = pc.Index(host=config.PINECONE_HOST)

# Initialize HF Inference Client for Embeddings
hf_client = InferenceClient(token=config.HF_TOKEN)

def get_query_embedding(text: str) -> list:
    """
    Generates embeddings for the search query using the Hugging Face Inference API.
    Model: sentence-transformers/all-mpnet-base-v2 (768 dimensions)
    """
    vector = hf_client.feature_extraction(
        text,
        model=config.EMBEDDING_MODEL_NAME
    )
    # Handle response format (ensure 1D list)
    if isinstance(vector, list) and len(vector) > 0 and isinstance(vector[0], list):
        return vector[0]
    return vector

def retrieve_similar_images(query_text: str, top_k: int = 3):
    """
    Embeds the query and searches Pinecone for the top_k most similar images.
    Returns a list of dictionaries with metadata and scores.
    """
    # 1. Embed the search query
    try:
        query_vector = get_query_embedding(query_text)
        # Ensure it's a list for serialization
        if hasattr(query_vector, "tolist"):
            query_vector = query_vector.tolist()
    except Exception as e:
        print(f"Error generating query embedding: {e}")
        return []
    
    # 2. Query Pinecone
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )
    
    # 3. Format the response
    retrieved_assets = []
    for match in results['matches']:
        metadata = match.get('metadata', {})
        retrieved_assets.append({
            "file_path": metadata.get('file_path'),
            "caption": metadata.get('caption'),
            "score": match.get('score')
        })
    
    return retrieved_assets

if __name__ == "__main__":
    # Get query from command line arguments or use default
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "a photo of a scenic view" # Default test query
    
    print(f"\n--- Searching for: '{query}' ---")
    
    try:
        results = retrieve_similar_images(query)
        
        if not results:
            print("No matching images found in the database.")
        else:
            for i, res in enumerate(results):
                print(f"\n[Result #{i+1}] (Similarity Score: {res['score']:.4f})")
                print(f"📁 Path: {res['file_path']}")
                # Print a snippet of the caption
                clean_caption = res['caption'].replace('\n', ' ')
                print(f"📝 Caption: {clean_caption[:150]}...")
    except Exception as e:
        print(f"An error occurred during retrieval: {e}")
