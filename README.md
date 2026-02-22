---
title: "Multi-modal RAG-driven Digital Asset Manager"
emoji: "🖼️"
sdk: "docker"
app_port: 7860
license: "mit"
---

# Multi-modal RAG-driven Digital Asset Manager

**An end-to-end engineering solution for semantic search and conversational reasoning over unstructured image libraries.**

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://bkqz-digital-asset-manager.hf.space)

---

## System Architecture & Rationale

This project addresses the challenge of retrieving unstructured visual data through a **decoupled RAG (Retrieval-Augmented Generation) architecture**. By separating heavy vision inference from low-latency vector search, the system remains scalable and responsive.

### High-Level Architecture

```mermaid
%%{init: { 
  'theme': 'base',
  'themeVariables': {
    'fontSize': '18px',
    'fontFamily': 'inter, sans-serif',
    'primaryColor': '#ffffff',
    'primaryBorderColor': '#1A237E',
    'lineColor': '#212121',
    'tertiaryColor': '#FAFAFA'
  },
  'flowchart': {
    'nodeSpacing': 50,
    'rankSpacing': 80,
    'padding': 20
  }
}}%%

flowchart LR
    %% High-Contrast Class Definitions
    classDef ingestion fill:#E3F2FD,stroke:#1565C0,stroke-width:4px,color:#0D47A1,font-weight:800;
    classDef storage fill:#FFFDE7,stroke:#F9A825,stroke-width:4px,color:#BF360C,font-weight:800;
    classDef query fill:#F3E5F5,stroke:#7B1FA2,stroke-width:4px,color:#4A148C,font-weight:800;

    subgraph INGESTION ["`INGESTION FLOW`"]
        direction TB
        IMG_UPLOAD[" Image Upload "] --> SUPA_STORE[" Supabase Storage "]
        SUPA_STORE --> LLAMA_VISION[" Llama 4 Vision Captioning "]
        LLAMA_VISION --> EMBED_GEN[" Generate Vector Embeddings "]
        EMBED_GEN -- " Caption + URL + Vector " --> PINECONE[(" Pinecone Index ")]
    end

    subgraph QUERY [" USER QUERY FLOW "]
        direction TB
        USER_INPUT[" Natural Language Query "] --> QUERY_EMBED[" Embed Search Query "]
        QUERY_EMBED -- Similarity Search --> PINECONE
        PINECONE -- " Top-K Matches " --> STREAMLIT_UI[" Streamlit UI Display "]
        STREAMLIT_UI <-- " Contextual Reasoning " --> LLAMA_REASON[" Llama 3.3 Reasoning "]
    end

    %% Interaction Link
    SUPA_STORE -. "`Public URI Source`" .-> PINECONE
    
    %% Applying Accessible Classes
    class IMG_UPLOAD,LLAMA_VISION,EMBED_GEN ingestion;
    class PINECONE,SUPA_STORE storage;
    class USER_INPUT,QUERY_EMBED,STREAMLIT_UI,LLAMA_REASON query;
```

### Engineering Decisions
*   **Vision Inference**: Utilizes **Llama-4-Scout via Groq's LPU™** technology. I chose Groq to achieve sub-second caption generation, which is typically the bottleneck in multimodal pipelines.
*   **Vector Orchestration**: Employs **Pinecone Serverless** for a 768-dimensional index. Pinecone was selected for its native support of metadata filtering and horizontal scaling.
*   **Persistent Storage**: **Supabase Storage** provides cloud-native asset hosting, ensuring that retrieved results are accessible via high-speed CDNs rather than local disk.

---

## Implementation Details

### 1. Asynchronous Ingestion Pipeline
The ingestion flow is designed as a linear ETL (Extract, Transform, Load) process:
- **Storage**: Binary data is persisted to a public Supabase bucket to generate a permanent URI.
- **Cognition**: The image URI (or base64) is passed to Llama-4-Scout to generate a semantic description.
- **Embedding**: The description is vectorized using `all-mpnet-base-v2`, mapping the image into a 768-dimensional latent space.
- **Sink**: The resulting vector, URI, and original caption are upserted into Pinecone.

### 2. Retrieval & Reasoning (RAG)
To move beyond simple search, I implemented a **Search-then-Reason** loop:
1.  **Similarity Search**: User queries are embedded in real-time and matched using Cosine Similarity.
2.  **Context Injection**: Top-$k$ matches are retrieved, and their captions are injected into a prompt as "Ground Truth" context.
3.  **Synthesis**: A Llama-3.3-70b model processes the query against this context, ensuring that the AI's responses are strictly grounded in the user's actual image library.

---

## Deployment & Local Setup

### Prerequisites
- Python 3.11+
- Docker
- API keys: Groq, Pinecone, Hugging Face, Supabase.

### Environment Configuration
Configure your `.env` file:

```bash
# Vector DB
PINECONE_API_KEY=your_key
PINECONE_HOST=your_host_url # Omit https://
PINECONE_INDEX_NAME=digital-asset-manager

# LLM & Embeddings
GROQ_API_KEY=your_key
HF_TOKEN=your_token

# Cloud Storage
SUPABASE_URL=your_project_url
SUPABASE_KEY=your_anon_key
SUPABASE_BUCKET_NAME=assets
```

### Containerized Execution
The project uses a **multi-stage Docker build** to minimize image size and improve security by running as a non-root user.

```bash
# Build & Run
docker build -t asset-manager:latest .
docker run -p 7860:7860 --env-file .env asset-manager:latest
```

---

## License
MIT License - Developed as a technical demonstration of multimodal RAG architectures.
