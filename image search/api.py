import os
import faiss
import torch
import numpy as np

from io import BytesIO
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Query
from sentence_transformers import SentenceTransformer

####################
# GLOBALS / SINGLETONS
####################
app = FastAPI()
index = None   # Will hold the FAISS index
model = None   # Will hold the CLIP model

####################
# STARTUP HOOK
####################
@app.on_event("startup")
def load_model_and_index():
    global index, model
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("clip-ViT-B-32", device=device)
    
    # Use the full/absolute path
    index_path = r"D:\Anand\Jsearch_ai\image search\fashion.index"
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index not found at '{index_path}'")
    
    index = faiss.read_index(index_path)
    print(f"[Startup] Index loaded from {index_path}. Total items in index: {index.ntotal}")

####################
# HELPER FUNCTION
####################
def encode_image_file(file: UploadFile) -> np.ndarray:
    """
    Reads an uploaded image file into memory, encodes it with CLIP, 
    and returns the embedding as a float32 numpy array.
    """
    # Load the file into a PIL Image
    pil_image = Image.open(BytesIO(file.file.read())).convert("RGB")
    
    # Encode the image with CLIP (on CPU or GPU depending on 'model' device)
    embedding = model.encode(pil_image)
    # FAISS prefers float32
    return embedding.astype(np.float32)

####################
# ROUTES / ENDPOINTS
####################

@app.post("/search-image")
async def search_image(file: UploadFile = File(...), top_k: int = 5):
    """
    Accept an uploaded image. Encode it with CLIP. 
    Perform a similarity search in the FAISS index. 
    Return the top_k product IDs and similarity scores.
    """
    # 1) Encode the incoming image
    query_vec = encode_image_file(file)

    # 2) Perform FAISS search
    #    Reshape to (1, dimensionality)
    query_vec = query_vec.reshape(1, -1)
    distances, ids = index.search(query_vec, top_k)
    
    # The result is arrays of shape (1, top_k).
    retrieved_ids = ids[0]
    retrieved_scores = distances[0]
    
    # 3) Format and return
    results = []
    for pid, score in zip(retrieved_ids, retrieved_scores):
        results.append({
            "product_id": int(pid),
            "similarity_score": float(score)  # convert from np.float32 to Python float
        })
    
    return {"results": results}
