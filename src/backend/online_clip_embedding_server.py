import os
from typing import List

import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL_PATH = os.getenv("EM2MEM_VLM2VEC_MODEL_PATH", "/home/tahagpt/models/clip-vit-base-patch32")
DEVICE = os.getenv("EM2MEM_VLM2VEC_DEVICE", "cpu")
HOST = os.getenv("EM2MEM_VLM2VEC_EMBED_HOST", "127.0.0.1")
PORT = int(os.getenv("EM2MEM_VLM2VEC_EMBED_PORT", "18091"))

app = FastAPI()

print(f"[clip_embedding_server] loading CLIP from {MODEL_PATH} on {DEVICE}")
model = CLIPModel.from_pretrained(MODEL_PATH).to(DEVICE)
processor = CLIPProcessor.from_pretrained(MODEL_PATH)
model.eval()
print("[clip_embedding_server] model loaded")


def _unwrap(output):
    return output.pooler_output if hasattr(output, "pooler_output") else output


def _normalize(vecs: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vecs, axis=1, keepdims=True)
    norm = np.maximum(norm, 1e-12)
    return vecs / norm


class ImageRequest(BaseModel):
    image_paths: List[str]
    normalize: bool = True


class TextRequest(BaseModel):
    texts: List[str]
    normalize: bool = True


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}


@app.post("/embed/images")
def embed_images(req: ImageRequest):
    images = [Image.open(p).convert("RGB") for p in req.image_paths]
    inputs = processor(images=images, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = _unwrap(model.get_image_features(**inputs))
    vecs = out.float().cpu().numpy()
    if req.normalize:
        vecs = _normalize(vecs)
    return {"embeddings": vecs.tolist()}


@app.post("/embed/texts")
def embed_texts(req: TextRequest):
    inputs = processor(text=req.texts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    with torch.no_grad():
        out = _unwrap(model.get_text_features(**inputs))
    vecs = out.float().cpu().numpy()
    if req.normalize:
        vecs = _normalize(vecs)
    return {"embeddings": vecs.tolist()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
