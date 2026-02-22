import os
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index = pc.Index(host=os.environ.get("PINECONE_HOST"))

# Check stats
stats = index.describe_index_stats()
print(f"Index stats: {stats}")

# Query one of the ingested IDs
# Note: I'll use the IDs from the previous output
test_id = "161fd66f-77fd-43de-9ab2-dc424fbeae2b"
res = index.fetch(ids=[test_id])
if test_id in res['vectors']:
    metadata = res['vectors'][test_id]['metadata']
    print(f"Successfully fetched vector {test_id}")
    print(f"Metadata: {metadata}")
else:
    print(f"Could not find vector {test_id}")
