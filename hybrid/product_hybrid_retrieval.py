import os
import re
from typing import List, TypedDict
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from hybrid.custom_sparse_retriever import CustomSparseRetriever
from utils.utils import load_config, initialize_embeddings, load_faiss_store
from logger.logger import get_logger
from elasticsearch import Elasticsearch
from langchain.retrievers import EnsembleRetriever
from pprint import pformat

logger = get_logger(__file__)

# -------------------------------------------------------------------
# 2) Define Pipeline State
# -------------------------------------------------------------------
class ProductState(TypedDict):
    question: str
    k: int
    context: List[Document]     # Documents retrieved from FAISS or ensemble retriever
    faiss_metadata: str         # Assembled metadata string
    llm_text: str               # Final LLM output text
    loop_step: int

# -------------------------------------------------------------------
# 4) Retrieval: Retrieve k Products Using an Ensemble Retriever
# -------------------------------------------------------------------
def retrieve_products(state: ProductState, faiss_store, es_client, bm25_index):
    """
    Retrieve up to k products using an ensemble retriever that combines dense and sparse retrieval.
    Logs only the metadata of the documents retrieved by the dense and sparse retrievers.
    """
    try:
        k = state.get("k", 200)
        question = state["question"]

        # Initialize dense and sparse retrievers.
        dense_retriever = faiss_store.as_retriever(search_kwargs={'k': k})
        sparse_retriever = CustomSparseRetriever(es_client, bm25_index, size=k)

        # Retrieve documents from the dense retriever and log metadata.
        dense_docs = dense_retriever.invoke(question)
        dense_metadata = [doc.metadata for doc in dense_docs]
        logger.info("==== DENSE RETRIEVER METADATA ====")
        logger.info("Retrieved %d documents:" % len(dense_metadata))
        #logger.info(pformat(dense_metadata))

        # Retrieve documents from the sparse retriever and log metadata.
        sparse_docs = sparse_retriever.invoke(question)
        sparse_metadata = [doc.metadata for doc in sparse_docs]
        logger.info("==== SPARSE RETRIEVER METADATA ====")
        logger.info("Retrieved %d documents:" % len(sparse_metadata))
        #logger.info(pformat(sparse_metadata))

        # Combine using the ensemble retriever.
        ensemble_retriever = EnsembleRetriever(
            retrievers=[dense_retriever, sparse_retriever],
            weights=[0.3, 0.7]
        )
        ensemble_docs = ensemble_retriever.invoke(question)
        ensemble_metadata = [doc.metadata for doc in ensemble_docs]
        logger.info("==== ENSEMBLE RETRIEVER OUTPUT ====")
        logger.info("Combined output (retrieved %d documents with k=%d):" % (len(ensemble_metadata), k))
        #logger.info(pformat(ensemble_metadata))

        return {"context": ensemble_docs}
    except Exception as e:
        logger.error("Error retrieving %d docs: %s" % (k, e), exc_info=True)
        return {"context": []}


def gather_faiss_metadata(state: ProductState) -> ProductState:
    """
    Reads all docs in `state["context"]` and assembles metadata lines into `faiss_metadata`,
    while preserving the existing state.
    """
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
    
    logger.info("FAISS metadata successfully gathered.")
    # Update the existing state rather than returning a new dict
    state["faiss_metadata"] = metadata_str
    return state


# -------------------------------------------------------------------
# 4) Node: Call LLM (Streaming)
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
    """
    Build a prompt from up to 25 documents, call the LLM, and store its output in state["llm_text"].
    """
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

# -------------------------------------------------------------------
# 5) Build Ensemble Pipeline: Dense + Sparse Retrieval
# -------------------------------------------------------------------
def build_ensemble_pipeline(faiss_store, es_client, bm25_index):
    """
    Build a pipeline that uses ensemble retrieval:
      START -> retrieve_products (with dense and sparse retrievers) -> gather_faiss_metadata (end)
    """
    graph_builder = StateGraph(ProductState)
    graph_builder.add_node("retrieve_products", lambda s: retrieve_products(s, faiss_store, es_client, bm25_index))
    graph_builder.add_node("gather_faiss_metadata", gather_faiss_metadata)
    
    graph_builder.add_edge(START, "retrieve_products")
    graph_builder.add_edge("retrieve_products", "gather_faiss_metadata")
    
    compiled_graph = graph_builder.compile()
    logger.info("Ensemble pipeline compiled.")
    return compiled_graph

# -------------------------------------------------------------------
# 6) Build Pipeline B: LLM-Only Pipeline
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
# 7) Main Execution (for local testing)
# -------------------------------------------------------------------
def main():
    try:
        # Load configuration, LLM, FAISS, and Elasticsearch
        this_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(this_dir, "..", "config.yaml")
        config = load_config(config_path)
        logger.info(f"Configuration loaded from {config_path}")
        
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
        
        embeddings = initialize_embeddings(config["output_dir"])
        faiss_store = load_faiss_store(config["product_store_path"], embeddings)
        logger.info(f"FAISS vector store loaded from: {config['product_store_path']}")
        
        es_url = config["elasticsearch_url"]
        bm25_index = config.get("product_index_name", "products")
        es_client = Elasticsearch(es_url)
        
        # Build the ensemble retrieval pipeline (dense + sparse)
        ensemble_pipeline = build_ensemble_pipeline(faiss_store, es_client, bm25_index)
        # Build the LLM pipeline
        llm_pipeline = build_llm_pipeline(llm)
        
        # Interactive Loop
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
            
            # Use the ensemble pipeline for retrieval
            ensemble_state = ensemble_pipeline.invoke(initial_state)
            ensemble_state = gather_faiss_metadata(ensemble_state)
            
            print("\n=== FAISS METADATA ===")
            print(ensemble_state.get("faiss_metadata", "No metadata found."))
            print("======================\n")
            
            # Use the LLM pipeline for generating recommendations
            final_llm_state = None
            print("=== STREAMING LLM TOKENS ===")
            for msg, metadata in llm_pipeline.stream(
                ensemble_state,
                stream_mode="messages"
            ):
                if metadata["langgraph_node"] == "call_llm_recommendation":
                    print(msg.content, end="", flush=True)
                if metadata.get("langgraph_phase") == "end":
                    final_llm_state = metadata["final_state"]
            print("\n=== END OF LLM STREAMING ===\n")
            
    except KeyboardInterrupt:
        logger.info("User interrupted. Exiting gracefully.")
    except Exception as e:
        logger.error(f"An error occurred in the main function: {e}", exc_info=True)
        raise e

if __name__ == "__main__":
    main()
