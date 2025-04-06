import yaml
import pandas as pd
import torch
import faiss
import numpy as np
import mlflow
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS  # Updated import
from logger.logger import get_logger
import os

# Create a logger for this module
logger = get_logger(__name__)

def load_config(config_path: str):
    """
    Load configuration from a YAML file.
    Args:
        config_path (str): Path to the configuration file.
    Returns:
        dict: Configuration dictionary.
    """
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration from {config_path}: {e}")
        raise e

def load_dataframe(df_path: str) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.
    Args:
        df_path (str): Path to the CSV file.
    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    try:
        df = pd.read_csv(df_path)
        logger.info(f"Dataframe loaded from {df_path} with {len(df)} records.")
        return df
    except Exception as e:
        logger.error(f"Failed to load dataframe from {df_path}: {e}")
        raise e
    
def initialize_mlflow_model(model_uri: str):
    try:
        # Set the tracking URI if not already set
        mlflow.set_tracking_uri("http://192.168.1.227:5000")
        
        # Load the model using MLflow's sentence_transformers flavor
        model = mlflow.sentence_transformers.load_model(model_uri)
        
        # Determine device and move model if necessary
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)  # Ensure the model is on the correct device
        
        logger.info(f"Model initialized using MLflow model at: {model_uri} on device: {device}")
        return model
    except Exception as e:
        logger.error(f"Failed to initialize model from {model_uri}: {e}")
        raise e
    
def initialize_embeddings(model_path: str):
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embeddings = HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={"device": device}
        )
        logger.info(f"Embeddings initialized using model at: {model_path} on device: {device}")
        return embeddings
    except Exception as e:
        logger.error(f"Failed to initialize embeddings from {model_path}: {e}")
        raise e
    

def add_documents_to_store(documents, vector_store, batch_size=1000):
    """
    Adds documents to the vector store in batches.
    """
    total = len(documents)
    for i in range(0, total, batch_size):
        batch = documents[i:i + batch_size]
        vector_store.add_documents(batch)
        # Removed per-batch logging as requested:
        # logger.info(f"Added documents {i+1} to {i+len(batch)} out of {total}")



# ----------------------------------------------------------
# 2) Build FAISS Vector Store (Custom Approach)
# ----------------------------------------------------------
   
def normalize_vector(vec):
    """
    Normalize a vector to unit length. Returns the original vector if its norm is zero.
    """
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm

def build_faiss_vector_store(embedding_function):
    """
    Initialize an empty FAISS store (IndexFlatIP) for cosine similarity and return it.
    Embeddings are normalized before being used in the index.
    """
    try:
        # Determine embedding dimension using a test query and normalize it
        test_emb = embedding_function.embed_query("test")
        test_emb = normalize_vector(np.array(test_emb))
        embedding_dim = len(test_emb)
        logger.info(f"Embedding dimension: {embedding_dim}")
    except Exception as e:
        logger.error(f"Error determining embedding dimension: {e}", exc_info=True)
        raise e

    try:
        # Create the FAISS index using inner product for cosine similarity.
        index = faiss.IndexFlatIP(embedding_dim)
        docstore = InMemoryDocstore()  # Ensure this is imported/defined in your code.
        vector_store = FAISS(
            embedding_function=embedding_function,
            index=index,
            docstore=docstore,
            normalize_L2=True,  # Ensure embeddings are normalized for cosine similarity
            index_to_docstore_id={}
        )
        logger.info(f"FAISS vector store initialized using IndexFlatIP (dimension={embedding_dim}).")
        return vector_store
    except Exception as e:
        logger.error(f"Error initializing FAISS store: {e}", exc_info=True)
        raise e

def load_faiss_store(store_path: str, embeddings):
    """
    Load a FAISS vector store from a local path.
    Args:
        store_path (str): Path to the FAISS vector store.
        embeddings: Embedding function used to initialize the vector store.
    Returns:
        FAISS: Loaded FAISS vector store.
    """
    try:
        logger.info(f"Loading FAISS vector store from: {store_path}")
        faiss_store = FAISS.load_local(
            folder_path=store_path,
            embeddings=embeddings,
            allow_dangerous_deserialization=True  # Required due to pickle usage
        )
        logger.info("FAISS vector store loaded successfully.")
        return faiss_store
    except Exception as e:
        logger.error(f"Failed to load FAISS vector store: {e}")
        raise e



def build_faiss_vector_store2(embedding_function, nlist=100):
    """
    Initialize a FAISS store using an IVFFlat index for cosine similarity.
    The index is built using normalized embeddings.
    Args:
        embedding_function: The embedding model.
        nlist (int): Number of clusters for IVFFlat.
    Returns:
        FAISS: A LangChain FAISS vector store.
    """
    try:
        # Determine embedding dimension using a test query and normalize it
        test_emb = embedding_function.embed_query("test")
        test_emb = normalize_vector(np.array(test_emb))
        embedding_dim = len(test_emb)
        logger.info(f"Embedding dimension: {embedding_dim}")
    except Exception as e:
        logger.error(f"Error determining embedding dimension: {e}", exc_info=True)
        raise e

    try:
        # Create a quantizer using inner product (for cosine similarity on normalized vectors)
        quantizer = faiss.IndexFlatIP(embedding_dim)
        # Build the IVFFlat index; note: IndexIVFFlat requires training before adding vectors.
        index = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist, faiss.METRIC_INNER_PRODUCT)
        logger.info(f"FAISS IVFFlat index created with nlist={nlist} and embedding dimension={embedding_dim}.")
        # Note: The index must be trained before adding vectors.
        docstore = InMemoryDocstore()
        vector_store = FAISS(
            embedding_function=embedding_function,
            index=index,
            docstore=docstore,
            index_to_docstore_id={}
        )
        logger.info("FAISS vector store initialized using IVFFlat for cosine similarity.")
        return vector_store
    except Exception as e:
        logger.error(f"Error initializing FAISS store: {e}", exc_info=True)
        raise e