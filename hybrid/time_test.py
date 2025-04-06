import os
import re
import time  # <-- ADDED for timing
from typing import List, TypedDict

# LangChain / LLM / Documents
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

# Hybrid / Ensemble
from hybrid.custom_sparse_retriever import CustomSparseRetriever
from langchain.retrievers import EnsembleRetriever

# Utilities
from utils.utils import load_config, initialize_embeddings, load_faiss_store
from logger.logger import get_logger

# Elasticsearch
from elasticsearch import Elasticsearch


# CrossEncoder for re-ranking
import torch
from sentence_transformers import CrossEncoder

# LangGraph
from langgraph.graph import START, StateGraph

logger = get_logger(__file__)

# -------------------------------------------------------------------
# 1) Define Pipeline State
# -------------------------------------------------------------------
class ProductState(TypedDict):
    question: str
    k: int
    context: List[Document]     # Documents retrieved from FAISS or ensemble retriever
    faiss_metadata: str         # Assembled metadata string
    llm_text: str               # Final LLM output text
    loop_step: int

# -------------------------------------------------------------------
# 2) Ensemble Retrieval: Dense + Sparse
# -------------------------------------------------------------------
def retrieve_products(state: ProductState, faiss_store, es_client, bm25_index):
    """
    Retrieve up to k products using an ensemble of dense + sparse retrieval.
    """
    start_time = time.perf_counter()  # <-- TIMING START
    try:
        k = state.get("k", 200)
        question = state["question"]

        # Initialize dense and sparse retrievers
        dense_retriever = faiss_store.as_retriever(search_kwargs={'k': k})
        sparse_retriever = CustomSparseRetriever(es_client, bm25_index, size=k)

        # Retrieve documents from the dense retriever and log metadata.
        dense_docs = dense_retriever.invoke(question)
        dense_metadata = [doc.metadata for doc in dense_docs]
        logger.info("==== DENSE RETRIEVER METADATA ====")
        logger.info("Retrieved %d documents." % len(dense_metadata))

        # Retrieve documents from the sparse retriever and log metadata.
        sparse_docs = sparse_retriever.invoke(question)
        sparse_metadata = [doc.metadata for doc in sparse_docs]
        logger.info("==== SPARSE RETRIEVER METADATA ====")
        logger.info("Retrieved %d documents." % len(sparse_metadata))

        # Combine using EnsembleRetriever
        ensemble_retriever = EnsembleRetriever(
            retrievers=[dense_retriever, sparse_retriever],
            weights=[0.3, 0.7]  # Adjust weights as needed
        )
        ensemble_docs = ensemble_retriever.invoke(question)
        ensemble_metadata = [doc.metadata for doc in ensemble_docs]
        logger.info("==== ENSEMBLE RETRIEVER OUTPUT ====")
        logger.info("Combined output (retrieved %d documents with k=%d).", len(ensemble_metadata), k)

        return {"context": ensemble_docs}
    except Exception as e:
        logger.error("Error retrieving %d docs: %s", k, e, exc_info=True)
        return {"context": []}
    finally:
        elapsed = time.perf_counter() - start_time  # <-- TIMING END
        logger.info(f"retrieve_products took {elapsed:.3f} seconds.")

# -------------------------------------------------------------------
# 3) Cross-Encoder Re-ranking
# -------------------------------------------------------------------
def cross_encoder_rerank(state: ProductState, cross_encoder) -> ProductState:
    start_time = time.perf_counter()  # <-- TIMING START
    try:
        docs = state.get("context", [])
        if not docs:
            logger.warning("No documents found for cross-encoder re-ranking.")
            return state

        query = state["question"]

        # Build input pairs for cross-encoder: (query, doc_text)
        pairs = [(query, doc.page_content) for doc in docs]

        # Compute scores
        scores = cross_encoder.predict(pairs)

        # Sort documents (descending) by cross-encoder score
        scored_docs = sorted(
            zip(docs, scores),
            key=lambda x: x[1],
            reverse=True
        )

        re_ranked_docs = [doc for doc, score in scored_docs]
        state["context"] = re_ranked_docs  # Update pipeline state with re-ranked docs

        logger.info("Cross-encoder re-ranking completed. Top doc now: %s", re_ranked_docs[0].metadata if re_ranked_docs else "None")
        return state
    finally:
        elapsed = time.perf_counter() - start_time  # <-- TIMING END
        logger.info(f"cross_encoder_rerank took {elapsed:.3f} seconds.")

# -------------------------------------------------------------------
# 4) Gather Metadata
# -------------------------------------------------------------------
def gather_metadata(state: ProductState) -> ProductState:
    start_time = time.perf_counter()  # <-- TIMING START
    try:
        docs = state.get("context", [])
        if not docs:
            logger.warning("No documents found to gather metadata.")
            state["faiss_metadata"] = "No documents retrieved; no metadata."
            return state
        
        lines = []
        for doc in docs:
            items = [f"{key}: {val}" for key, val in doc.metadata.items()]
            lines.append(", ".join(items))
        metadata_str = "\n".join(lines)
        
        logger.info("metadata successfully gathered.")
        state["faiss_metadata"] = metadata_str
        return state
    finally:
        elapsed = time.perf_counter() - start_time  # <-- TIMING END
        logger.info(f"gather_metadata took {elapsed:.3f} seconds.")

# -------------------------------------------------------------------
# 5) LLM Call (Streaming)
# -------------------------------------------------------------------
custom_recommendation_prompt = PromptTemplate(
    template="""You are an intelligent assistant that summarizes the product(s) based on the user's query from the available products.

**Instructions:**
- Generate a concise product(s) info for all the available product(s) in a paragraph, naturally integrating specific product details such as product name, location, and variant.
- Suggest related products that the user might find interesting within the context.
- Do not introduce or suggest any products that are not mentioned in the context.
- If the user's query is vague or general, gently prompt the user by suggesting specific product names or categories to explore further.
- Encourage user interaction by inviting them to refine your query or ask follow-up questions.

Context:
{context}

User Query:
{question}

Recommendation:
""",
    input_variables=["context", "question"]
)

def call_llm_recommendation(state: ProductState, llm) -> ProductState:
    start_time = time.perf_counter()  # <-- TIMING START (non-streaming portion)
    docs = state.get("context", [])
    if not docs:
        logger.warning("No documents found for LLM recommendation.")
        return {"llm_text": "No documents for LLM."}
    
    top25 = docs[:25]
    formatted_context = "\n".join(
        f"Product ID: {doc.metadata.get('product_id', 'Unknown')}, "
        f"Product Name: {doc.metadata.get('product_name', 'Unknown')}, "
        f"Supplier: {doc.metadata.get('Supplier', 'Unknown')}, "
        f"City: {doc.metadata.get('city', 'Unknown')}, "
        f"State: {doc.metadata.get('state', 'Unknown')}, "
        f"Country: {doc.metadata.get('country', 'Unknown')}, "
        f"Variant: {doc.metadata.get('variant', 'Unknown')}, "
        f"Category: {doc.metadata.get('category', 'Unknown')}"
        for doc in top25
    )
    
    prompt = custom_recommendation_prompt.format(
        context=formatted_context,
        question=state["question"]
    )
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        llm_text = response.content
        logger.info("LLM call completed, storing in 'llm_text'.")
        return {"llm_text": llm_text}
    except Exception as e:
        logger.error("LLM call failed.", exc_info=True)
        return {"llm_text": "LLM call failed; no recommendation."}
    finally:
        elapsed = time.perf_counter() - start_time  # <-- TIMING END
        logger.info(f"call_llm_recommendation (non-stream) took {elapsed:.3f} seconds.")

# -------------------------------------------------------------------
# 6) Build the Ensemble + Cross-Encoder Pipeline
# -------------------------------------------------------------------
def build_ensemble_pipeline(faiss_store, es_client, bm25_index, cross_encoder):
    """
    Build a pipeline that:
      1) retrieves documents using the ensemble of dense + sparse,
      2) re-ranks them with the cross-encoder,
      3) assembles metadata,
      4) ends.
    """
    graph_builder = StateGraph(ProductState)

    # Node: Ensemble retrieval
    graph_builder.add_node(
        "retrieve_products",
        lambda s: retrieve_products(s, faiss_store, es_client, bm25_index)
    )

    # Node: Cross-encoder re-rank
    graph_builder.add_node(
        "cross_encoder_rerank",
        lambda s: cross_encoder_rerank(s, cross_encoder)
    )

    # Node: Gather metadata
    graph_builder.add_node("gather_metadata", gather_metadata)

    # Define edges
    graph_builder.add_edge(START, "retrieve_products")
    graph_builder.add_edge("retrieve_products", "cross_encoder_rerank")
    graph_builder.add_edge("cross_encoder_rerank", "gather_metadata")

    compiled_graph = graph_builder.compile()
    logger.info("Ensemble + Cross-Encoder pipeline compiled.")
    return compiled_graph

# -------------------------------------------------------------------
# 7) Build the LLM-Only Pipeline
# -------------------------------------------------------------------
def build_llm_pipeline(llm):
    """
    Build a pipeline that calls the LLM:
      START -> call_llm_recommendation (end)
    """
    graph_builder = StateGraph(ProductState)
    graph_builder.add_node("call_llm_recommendation", lambda s: call_llm_recommendation(s, llm))
    graph_builder.add_edge(START, "call_llm_recommendation")
    compiled_graph = graph_builder.compile()
    logger.info("LLM-only pipeline compiled.")
    return compiled_graph

# -------------------------------------------------------------------
# 8) Main Execution (for local testing)
# -------------------------------------------------------------------
def main():
    try:
        # -----------------------------------------------------------------
        # A) Load configuration, LLM, FAISS, and Elasticsearch
        # -----------------------------------------------------------------
        this_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(this_dir, "..", "config.yaml")
        config = load_config(config_path)
        logger.info(f"Configuration loaded from {config_path}")
        
        # Detect device (CPU or GPU)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # Your LLM (e.g., ChatOllama)
        llm = ChatOllama(
            model="llama3.1:8b",
            stream=True,
            temperature=0.1,
            top_k=40,
            mirostat=0,
            repeat_penalty=1.2,
            num_ctx=4096,
            num_predict=1000,
            seed=42,
        )
        
        # Initialize embeddings for FAISS
        embeddings = initialize_embeddings(config["output_dir"])
        faiss_store = load_faiss_store(config["product_store_path"], embeddings)
        logger.info(f"FAISS vector store loaded from: {config['product_store_path']}")

        # Initialize Elasticsearch client
        es_url = config["elasticsearch_url"]
        bm25_index = config.get("product_index_name", "products")
        es_client = Elasticsearch(es_url)

        # -----------------------------------------------------------------
        # B) Initialize Cross-Encoder from config
        # -----------------------------------------------------------------
        cross_encoder = CrossEncoder(config["base_cross_encoder"], device=device)
        logger.info(f"Cross-encoder loaded from {config['base_cross_encoder']} on device {device}")

        # -----------------------------------------------------------------
        # C) Build Pipelines
        # -----------------------------------------------------------------
        # 1) Ensemble + Cross-Encoder
        ensemble_pipeline = build_ensemble_pipeline(
            faiss_store=faiss_store,
            es_client=es_client,
            bm25_index=bm25_index,
            cross_encoder=cross_encoder
        )

        # 2) LLM-only pipeline
        llm_pipeline = build_llm_pipeline(llm)

        # -----------------------------------------------------------------
        # D) Interactive Loop
        # -----------------------------------------------------------------
        while True:
            query = input("\nEnter your product search query (or type 'exit' to quit): ").strip()
            if query.lower() in ["exit", "quit"]:
                logger.info("Exiting.")
                break
            
            k_str = input("Enter number of documents to retrieve (k): ").strip()
            try:
                k_val = int(k_str)
            except ValueError:
                logger.warning("Invalid k; using default=200.")
                k_val = 200
            
            initial_state: ProductState = {
                "question": query,
                "k": k_val,
                "context": [],
                "faiss_metadata": "",
                "llm_text": "",
                "loop_step": 0,
            }
            
            # 1) Ensemble retrieval + Cross-Encoder re-rank
            ensemble_state = ensemble_pipeline.invoke(initial_state)

            # Print the final metadata after re-ranking
            print("\n=== FAISS METADATA (after cross-encoder re-rank) ===")
            print(ensemble_state.get("faiss_metadata", "No metadata found."))
            print("====================================================\n")
            
            # 2) Use the LLM pipeline for generating recommendations
            final_llm_state = None
            print("=== STREAMING LLM TOKENS ===")
            start_llm_streaming = time.perf_counter()  # <-- TIMING START for streaming

            for msg, metadata in llm_pipeline.stream(ensemble_state, stream_mode="messages"):
                if metadata["langgraph_node"] == "call_llm_recommendation":
                    # Stream the tokens to console
                    print(msg.content, end="", flush=True)
                
                if metadata.get("langgraph_phase") == "end":
                    final_llm_state = metadata["final_state"]

            end_llm_streaming = time.perf_counter()  # <-- TIMING END
            llm_stream_duration = end_llm_streaming - start_llm_streaming
            logger.info(f"LLM streaming duration: {llm_stream_duration:.3f} seconds.")
            print("\n=== END OF LLM STREAMING ===\n")

    except KeyboardInterrupt:
        logger.info("User interrupted. Exiting gracefully.")
    except Exception as e:
        logger.error(f"An error occurred in the main function: {e}", exc_info=True)
        raise e

if __name__ == "__main__":
    main()
