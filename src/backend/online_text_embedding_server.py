import os
from typing import List

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

MODEL_PATH = os.getenv("EM2MEM_TEXT_EMBED_MODEL_PATH", "/home/tahagpt/models/minilm-embed")
HOST = os.getenv("EM2MEM_TEXT_EMBED_HOST", "127.0.0.1")
PORT = int(os.getenv("EM2MEM_TEXT_EMBED_PORT", "18096"))

app = FastAPI()

print(f"[text_embedding_server] loading model from {MODEL_PATH}")
model = SentenceTransformer(MODEL_PATH, device="cpu")
print("[text_embedding_server] model loaded")


def _normalize(vecs: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vecs, axis=1, keepdims=True)
    norm = np.maximum(norm, 1e-12)
    return vecs / norm


class TextRequest(BaseModel):
    texts: List[str]
    batch_size: int = 256
    normalize: bool = False


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}


@app.post("/embed/texts")
def embed_texts(req: TextRequest):
    vecs = model.encode(req.texts, batch_size=req.batch_size, convert_to_numpy=True)
    vecs = vecs.astype("float32")
    if req.normalize:
        vecs = _normalize(vecs)
    return {"embeddings": vecs.tolist()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
