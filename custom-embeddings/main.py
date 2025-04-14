from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional, Union
import numpy
import uvicorn
import os
from dotenv import load_dotenv

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
    print("Warning: HF_TOKEN environment variable not set. Some models may not be accessible.")

model_name = "redis/langcache-embed-v1"
# Pass token to SentenceTransformer if available
model = SentenceTransformer(model_name, use_auth_token=hf_token)
embedding_dimensions = model.get_sentence_embedding_dimension()

# Print the embedding dimensions for configuration
print(f"Model: {model_name}, Embedding Dimensions: {embedding_dimensions}")


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

        if isinstance(request.input, str):
            texts = [request.input]
        else:
            texts = request.input

        embeddings = model.encode(texts, normalize_embeddings=True)

        data = []
        for i, embedding in enumerate(embeddings):
            embedding_list = embedding.astype(numpy.float32).tolist()
            data.append(EmbeddingData(embedding=embedding_list, index=i))

        total_tokens = sum(len(text.split()) *
                           1.3 for text in texts)

        return EmbeddingsResponse(
            data=data,
            usage={
                "prompt_tokens": int(total_tokens),
                "total_tokens": int(total_tokens),
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {
        "message": "Embeddings API is running",
        "model": model_name,
        "dimensions": embedding_dimensions
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=11434, reload=True)
