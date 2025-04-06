import os
import jwt
import datetime
from contextlib import asynccontextmanager
from typing import Optional, Dict

from fastapi import FastAPI, HTTPException, status, Depends
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel

# -----------------------------
# Domain-Specific Imports
# -----------------------------
# Replace these with your own pipeline logic
from semantic.product_retrieval import build_product_graph, ProductState
from semantic.service_retrieval import build_service_graph, ServiceState
from semantic.store_retrieval import build_store_graph, StoreState

# Common utilities
from utils.utils import load_config, initialize_embeddings, load_faiss_store
from utils.encrypt import aes_encrypt, aes_decrypt  # <-- Encryption functions added
from langchain_ollama import ChatOllama
from logger.logger import get_logger

logger = get_logger(__name__)

# -----------------------------
# JWT Configuration
# -----------------------------
SECRET_KEY = "REPLACE_THIS_WITH_A_LONG_RANDOM_SECRET"
ALGORITHM = "HS256"

# Token Lifetimes
ACCESS_TOKEN_EXPIRE_MINUTES = 15   # short-lived
REFRESH_TOKEN_EXPIRE_DAYS = 7      # longer-lived

# Simple user store
FAKE_USERS_DB = {
    "alice": {"username": "alice", "password": "wonderland123"},
    "bob":   {"username": "bob",   "password": "builder123"},
}

# -----------------------------
# JWT Helper Functions
# -----------------------------
def create_token(data: dict, expires_delta: datetime.timedelta) -> str:
    """
    General function to create a JWT with an expiration.
    `data` should include at least {"sub": username, "token_type": "access"|"refresh"}.
    """
    to_encode = data.copy()
    expire = datetime.datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(token: str, expected_token_type: str) -> Dict:
    """
    Verifies a JWT is valid, not expired, and has the correct token_type in its payload.
    Returns the decoded payload if valid; raises HTTP 401 if invalid.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("token_type") != expected_token_type:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid token type. Expected '{expected_token_type}'."
            )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired.")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token.")

# -----------------------------
# OAuth2 Setup
# -----------------------------
# We'll use password flow to get tokens:
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def authenticate_user(username: str, password: str) -> bool:
    """
    Verify user credentials against FAKE_USERS_DB (in-memory).
    In production, you'd verify hashed passwords from a real DB.
    """
    user = FAKE_USERS_DB.get(username.lower())
    if not user or user["password"] != password:
        return False
    return True

def get_current_user(access_token: str = Depends(oauth2_scheme)):
    """
    1) Extract Bearer token from Authorization header,
    2) Verify it as 'access' token,
    3) Return username if valid.
    """
    payload = verify_token(access_token, expected_token_type="access")
    username = payload.get("sub")
    if not username or username.lower() not in FAKE_USERS_DB:
        raise HTTPException(status_code=401, detail="Invalid token payload or user not found.")
    return username

# -----------------------------
# Request Models
# -----------------------------
class RefreshTokenRequest(BaseModel):
    refresh_token: str

class SearchRequest(BaseModel):
    query: str
    k: Optional[int] = 200

# -----------------------------
# Global Pipeline References
# -----------------------------
PRODUCT_GRAPH = None
SERVICE_GRAPH = None
STORE_GRAPH = None
SHARED_LLM = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup/Shutdown events for pipeline initialization.
    """
    global PRODUCT_GRAPH, SERVICE_GRAPH, STORE_GRAPH, SHARED_LLM
    try:
        # 1) Load configuration
        this_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(this_dir, "..", "config.yaml")
        config = load_config(config_path)
        logger.info(f"Configuration loaded from {config_path}")

        # 2) Create ChatOllama instance
        local_llm = "llama3.1:8b"
        SHARED_LLM = ChatOllama(model=local_llm, temperature=0, base_url="http://host.docker.internal:11434")

        # 3) Initialize embeddings
        embeddings = initialize_embeddings(config["output_dir"])

        # 4) Build each pipeline
        product_faiss = load_faiss_store(config["product_store_path"], embeddings)
        PRODUCT_GRAPH = build_product_graph(product_faiss, SHARED_LLM)
        logger.info("Product pipeline loaded successfully.")

        service_faiss = load_faiss_store(config["service_store_path"], embeddings)
        SERVICE_GRAPH = build_service_graph(service_faiss, SHARED_LLM)
        logger.info("Service pipeline loaded successfully.")

        store_faiss = load_faiss_store(config["store_v_store_path"], embeddings)
        STORE_GRAPH = build_store_graph(store_faiss, SHARED_LLM)
        logger.info("Store pipeline loaded successfully.")

        logger.info("All pipelines initialized with one shared LLM.")
    except Exception as e:
        logger.error(f"Error during startup initialization: {e}", exc_info=True)
        raise

    yield
    logger.info("Shutting down...")

app = FastAPI(lifespan=lifespan)

# -----------------------------
# 1) /token: Obtain Tokens
# -----------------------------
@app.post("/token")
def login_for_tokens(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Standard OAuth2 'password' flow endpoint.
    Takes form fields: username, password, (optionally grant_type=password).
    Returns both ACCESS and REFRESH tokens if credentials are correct.
    """
    if not authenticate_user(form_data.username, form_data.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    # Create short-lived Access Token
    access_payload = {"sub": form_data.username, "token_type": "access"}
    access_token = create_token(access_payload, datetime.timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))

    # Create longer-lived Refresh Token
    refresh_payload = {"sub": form_data.username, "token_type": "refresh"}
    refresh_token = create_token(refresh_payload, datetime.timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS))

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "access_expires_in_minutes": ACCESS_TOKEN_EXPIRE_MINUTES,
        "refresh_expires_in_days": REFRESH_TOKEN_EXPIRE_DAYS
    }

# -----------------------------
# 2) /refresh: Get New Access Token
# -----------------------------
@app.post("/refresh")
def refresh_access_token(req: RefreshTokenRequest):
    """
    Accepts a valid refresh token, returns a NEW short-lived access token.
    """
    payload = verify_token(req.refresh_token, expected_token_type="refresh")
    username = payload.get("sub")
    if not username or username.lower() not in FAKE_USERS_DB:
        raise HTTPException(status_code=401, detail="Invalid refresh token payload.")
    
    # Create a new Access Token
    new_access_payload = {"sub": username, "token_type": "access"}
    new_access_token = create_token(new_access_payload, datetime.timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))

    return {
        "access_token": new_access_token,
        "token_type": "bearer",
        "access_expires_in_minutes": ACCESS_TOKEN_EXPIRE_MINUTES
    }

# -----------------------------
# 3) Protected Endpoints
# -----------------------------
@app.post("/product_search")
def product_search(req: SearchRequest, current_user: str = Depends(get_current_user)):
    """
    Protected: must have a valid Access Token in Bearer auth.
    Decrypts the incoming query, processes it via the product pipeline,
    and returns an encrypted list of re-ranked product IDs.
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query text cannot be empty.")
    if not PRODUCT_GRAPH:
        raise HTTPException(status_code=500, detail="Product pipeline not initialized.")

    try:
        # Decrypt the incoming query
        decrypted_query = aes_decrypt(req.query)
        logger.info(f"Decrypted product query: {decrypted_query}")
    except Exception as e:
        logger.error("Error decrypting product query.", exc_info=True)
        raise HTTPException(status_code=400, detail="Invalid encrypted query.")

    # Build the initial state for the product pipeline
    initial_state: ProductState = {
        "question": decrypted_query,
        "k": req.k,
        "context": [],
        "final_ids_output": "",
        "loop_step": 0,
    }
    final_state = PRODUCT_GRAPH.invoke(initial_state)
    final_ids = final_state.get("final_ids_output", "No response provided.")

    # Encrypt the output before returning
    encrypted_result = aes_encrypt(final_ids)
    return {
        "domain": "Product",
        "user": current_user,
        "final_ids": encrypted_result
    }

