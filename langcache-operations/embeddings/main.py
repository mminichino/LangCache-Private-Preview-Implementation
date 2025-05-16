from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional, Union
import numpy
import uvicorn
import os
import sys
import logging
from dotenv import load_dotenv

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

app = FastAPI(
    title="Embeddings API",
    description="OpenAI-compatible API for generating embeddings using sentence-transformers",
    version="0.0.1",
)

# Get Hugging Face token from environment variable
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    logger.error("ERROR: HF_TOKEN environment variable not set.")
    raise ValueError("HF_TOKEN environment variable is required")

logger.info(f"HF_TOKEN is set, length: {len(hf_token)}")

# Load the redis/langcache-embed-v1 model or fallback to a default model
try:
    model_name = "redis/langcache-embed-v1"
    logger.info(f"Attempting to load model: {model_name}")

    # Print the token prefix to debug (don't log the full token for security reasons)
    logger.info(f"Using HF token starting with: {hf_token[:4]}...")

    try:
        # First try to load the redis/langcache-embed-v1 model
        model = SentenceTransformer(model_name, use_auth_token=hf_token)
        embedding_dimensions = model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded successfully: {model_name}, Embedding Dimensions: {embedding_dimensions}")
    except Exception as e:
        # If that fails, fall back to a more compatible model
        fallback_model = "all-MiniLM-L6-v2"
        logger.warning(f"Failed to load {model_name} model: {str(e)}")
        logger.info(f"Falling back to {fallback_model} model")

        model = SentenceTransformer(fallback_model)
        embedding_dimensions = model.get_sentence_embedding_dimension()
        model_name = fallback_model  # Update the model name to reflect what we're actually using
        logger.info(f"Fallback model loaded successfully: {model_name}, Embedding Dimensions: {embedding_dimensions}")
except Exception as e:
    error_msg = f"Failed to load any embedding model: {str(e)}"
    logger.error(error_msg)
    raise RuntimeError(error_msg)

class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    encoding_format: Optional[str] = "float"


class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int


class EmbeddingsResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    usage: Dict[str, int]


@app.post("/v1/embeddings", response_model=EmbeddingsResponse)
async def create_embeddings(request: EmbeddingRequest):
    try:
        logger.info(f"Processing embedding request with {len(request.input) if isinstance(request.input, list) else 1} inputs")

        if isinstance(request.input, str):
            texts = [request.input]
        else:
            texts = request.input

        embeddings = model.encode(texts, normalize_embeddings=True)

        data = []
        for i, embedding in enumerate(embeddings):
            embedding_list = embedding.astype(numpy.float32).tolist()
            data.append(EmbeddingData(embedding=embedding_list, index=i))

        total_tokens = sum(len(text.split()) * 1.3 for text in texts)

        return EmbeddingsResponse(
            data=data,
            usage={
                "prompt_tokens": int(total_tokens),
                "total_tokens": int(total_tokens),
            }
        )

    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {
        "message": "Embeddings API is running",
        "model": model_name,
        "dimensions": embedding_dimensions
    }


@app.get("/health")
async def health():
    """Health check endpoint for Cloud Run"""
    return {"status": "healthy", "model": model_name}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    logger.info(f"Starting server on port {port}")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
