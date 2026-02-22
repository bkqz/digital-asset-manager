import os
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = os.environ.get("PINECONE_INDEX_NAME")

try:
    index_description = pc.describe_index(index_name)
    print(f"Index Name: {index_description.name}")
    print(f"Host: {index_description.host}")
    
    current_host = os.environ.get("PINECONE_HOST")
    if current_host and index_description.host not in current_host:
        print(f"\nWARNING: Host mismatch!")
        print(f"Expected: {index_description.host}")
        print(f"Current in .env: {current_host}")
except Exception as e:
    print(f"Error: {e}")
