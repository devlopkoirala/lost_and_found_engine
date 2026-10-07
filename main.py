import io
import json
import os
from contextlib import asynccontextmanager
from typing import List

import faiss
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# Global state holders for in-memory resources
ML_MODELS = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager: Runs BEFORE the server starts receiving requests.
    Loads CLIP model and FAISS index into memory once.
    """
    print("\n--- Server Boot: Loading ML Assets into RAM ---")

    # 1. Load CLIP Model
    print("Loading CLIP ViT-B-32 model...")
    ML_MODELS["clip"] = SentenceTransformer("clip-ViT-B-32")

    # 2. Paths to indexed vectors & metadata
    embeddings_path = "data/embeddings.npy"
    paths_file = "data/image_paths.npy"

    if not os.path.exists(embeddings_path) or not os.path.exists(paths_file):
        raise FileNotFoundError(
            "Embeddings or image paths missing! Run extract_embeddings.py first."
        )

    # 3. Load vectors and paths from disk
    embeddings = np.load(embeddings_path).astype(np.float32)
    image_paths = np.load(paths_file).tolist()

    # 4. Build in-memory FAISS Index (Flat Inner Product for Cosine Similarity)
    dimension = embeddings.shape[1]  # 512 for ViT-B-32
    faiss.normalize_L2(embeddings)  # Normalize for cosine similarity

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    ML_MODELS["faiss_index"] = index
    ML_MODELS["image_paths"] = image_paths

    print(f"FAISS Index ready with {index.ntotal} catalog items.")
    print("--- Server Ready for Queries ---\n")

    yield  # Server runs and handles requests here

    # Cleanup when server shuts down
    ML_MODELS.clear()
    print("Cleaned up ML models from memory.")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Lost & Found Similarity Engine API",
    description="Search catalog for lost items using visual CLIP embeddings and FAISS",
    version="1.0.0",
    lifespan=lifespan,
)


# Response Schema (Pydantic models)
class MatchItem(BaseModel):
    image_path: str
    similarity_score: float


class SearchResponse(BaseModel):
    query_status: str
    total_results: int
    matches: List[MatchItem]


@app.get("/")
def health_check():
    """Simple endpoint to verify server is alive."""
    return {"status": "online", "catalog_items": ML_MODELS["faiss_index"].ntotal}


@app.post("/search", response_model=SearchResponse)
async def search_lost_item(
    file: UploadFile = File(...),
    top_k: int = 5
):
    """
    Accepts an uploaded photo of a 'lost' item and returns top-K visually matching found items.
    """
    # 1. Validate uploaded file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File uploaded must be an image.")

    try:
        # 2. Read raw file bytes into PIL Image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # 3. Extract embedding using global CLIP model
        clip_model = ML_MODELS["clip"]
        query_vector = clip_model.encode(image, convert_to_numpy=True).astype(np.float32)

        # 4. Reshape and L2 normalize query vector
        query_vector = np.expand_dims(query_vector, axis=0)
        faiss.normalize_L2(query_vector)

        # 5. Query FAISS index
        faiss_index = ML_MODELS["faiss_index"]
        image_paths = ML_MODELS["image_paths"]

        # Limit top_k to total items in index if top_k > catalog size
        k = min(top_k, faiss_index.ntotal)
        distances, indices = faiss_index.search(query_vector, k)

        # 6. Format JSON response
        matches = []
        for score, idx in zip(distances[0], indices[0]):
            if idx != -1:  # -1 indicates no match found
                matches.append(
                    MatchItem(
                        image_path=image_paths[idx],
                        similarity_score=float(score)
                    )
                )

        return SearchResponse(
            query_status="success",
            total_results=len(matches),
            matches=matches
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing search: {str(e)}")