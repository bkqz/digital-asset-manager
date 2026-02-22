---
title: "Multi-modal RAG-driven Digital Asset Manager"
emoji: "🖼️"
sdk: "docker"
app_port: 7860
license: "mit"
---

# Multi-modal RAG-driven Digital Asset Manager

An engineered solution for indexing, searching, and reasoning over unstructured visual data. This system implements a Retrieval-Augmented Generation (RAG) architecture that translates visual content into a high-dimensional latent space for semantic search.

## 🏗️ System Architecture

The application is built on a decoupled architecture, separating high-compute vision tasks from low-latency vector retrieval.

### Core Components
*   **Vision Engine**: Llama-4-Scout (via Groq) for asynchronous image-to-text synthesis.
*   **Vector Database**: Pinecone Serverless (768-dimensional index) utilizing cosine similarity for retrieval.
*   **Embedding Model**: `all-mpnet-base-v2` via Hugging Face Inference API for standardized vector representation.
*   **Blob Storage**: Supabase Storage for persistent, cloud-native asset hosting.
*   **Reasoning Layer**: Llama-3.3-70b (via Groq) for synthesizing natural language responses based on retrieved context.

---

## 🛠️ Data Pipelines

### 1. Ingestion Pipeline (ETL)
The ingestion flow transforms raw binary data into searchable metadata:
1.  **Object Persistence**: Local uploads are persisted to a public Supabase bucket.
2.  **Multimodal Inference**: The system generates a dense text description using Groq-accelerated Llama 4 Vision.
3.  **Vectorization**: Captions are embedded into a 768-dim vector.
4.  **Index Upsert**: The vector is stored in Pinecone alongside the Supabase public URL and generated metadata.

### 2. Retrieval & Reasoning Pipeline (Query)
The query flow implements a "Search-then-Reason" pattern:
1.  **Similarity Search**: User text queries are embedded and matched against the Pinecone index.
2.  **Context Injection**: The top-$k$ relevant image URLs and captions are retrieved to form the RAG context.
3.  **Synthetic Response**: The Reasoning Layer processes the query against the retrieved context to provide accurate, multi-turn dialogue regarding the assets.

---

## 🚀 Deployment & Local Development

### Prerequisites
*   Python 3.11+
*   Docker (for containerized deployment)
*   API keys for Groq, Pinecone, Hugging Face, and Supabase.

### Environment Configuration
Create a `.env` file in the root directory:

```bash
# Pinecone Configuration
PINECONE_API_KEY=your_key
PINECONE_HOST=your_index_host_url
PINECONE_INDEX_NAME=digital-asset-manager

# Inference APIs
GROQ_API_KEY=your_key
HF_TOKEN=your_token

# Cloud Storage (Supabase)
SUPABASE_URL=your_project_url
SUPABASE_KEY=your_anon_key
SUPABASE_BUCKET_NAME=assets
```

### Docker Execution
To run the production-grade multi-stage build:

```bash
# Build the image
docker build -t asset-manager:latest .

# Run the container
docker run -p 7860:7860 --env-file .env asset-manager:latest
```

The application will be accessible at `http://localhost:7860`.

---

## ⚖️ License
This project is licensed under the MIT License.
