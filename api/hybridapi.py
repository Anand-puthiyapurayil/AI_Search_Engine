import uuid
import os
import torch
from typing import Optional, Dict
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
from elasticsearch import Elasticsearch

from hybrid.product_hybrid_rerank_retrieval import (
    build_ensemble_pipeline,
    build_llm_pipeline,
    ProductState
)
from sentence_transformers import CrossEncoder
from langchain_ollama import ChatOllama

from utils.utils import load_config, initialize_embeddings, load_faiss_store
from logger.logger import get_logger

logger = get_logger(__name__)

# Globals for pipelines and session state storage
ENSEMBLE_PIPELINE = None
LLM_PIPELINE = None
SESSION_STATES: Dict[str, ProductState] = {}

# Pydantic models
class SearchRequest(BaseModel):
    query: str
    k: Optional[int] = 200

class LLMRequest(BaseModel):
    session_id: str  # use this to refer to state from first call

@asynccontextmanager
async def lifespan(app: FastAPI):
    global ENSEMBLE_PIPELINE, LLM_PIPELINE
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "..", "config.yaml")
    config = load_config(config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cross_encoder = CrossEncoder(config["base_cross_encoder"], device=device)
    embeddings = initialize_embeddings(config["output_dir"])
    faiss_store = load_faiss_store(config["product_store_path"], embeddings)
    es_client = Elasticsearch(config["elasticsearch_url"])

    llm = ChatOllama(
        model="llama3.1:8b",
        stream=True,
        temperature=0.1,
        top_k=40,
        repeat_penalty=1.2,
        num_ctx=4096,
        num_predict=1000,
        seed=42,
    )

    ENSEMBLE_PIPELINE = build_ensemble_pipeline(
        faiss_store, es_client, config.get("product_index_name", "products"), cross_encoder
    )

    LLM_PIPELINE = build_llm_pipeline(llm)

    logger.info("Pipelines initialized.")
    yield

app = FastAPI(lifespan=lifespan)

@app.post("/product_search")
def product_search(req: SearchRequest):
    global SESSION_STATES

    session_id = str(uuid.uuid4())  # generate unique session id
    initial_state: ProductState = {
        "question": req.query,
        "k": req.k,
        "context": [],
        "faiss_metadata": "",
        "llm_text": "",
        "loop_step": 0,
    }

    final_state = ENSEMBLE_PIPELINE.invoke(initial_state)

    # Store final state for use in LLM call later
    SESSION_STATES[session_id] = final_state

    product_data = final_state.get("faiss_metadata", "No products found")

    return {
        "session_id": session_id,  # client uses this in the next request
        "product_data": product_data
    }

@app.post("/product_search_llm")
def product_search_llm(req: LLMRequest):
    global SESSION_STATES

    state = SESSION_STATES.get(req.session_id)
    if state is None:
        raise HTTPException(status_code=400, detail="Invalid or expired session_id")

    def token_generator():
        try:
            for msg, metadata in LLM_PIPELINE.stream(state, stream_mode="messages"):
                if metadata["langgraph_node"] == "call_llm_recommendation":
                    if msg.content:
                        yield msg.content
        except Exception as e:
            logger.error(f"LLM streaming error: {e}", exc_info=True)
            yield f"\n[Error: {str(e)}]"

    return StreamingResponse(token_generator(), media_type="text/plain")
