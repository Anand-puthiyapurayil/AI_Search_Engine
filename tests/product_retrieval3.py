import os
import re
import json
import numpy as np
from typing import List, TypedDict
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from utils.utils import load_config, initialize_embeddings, load_faiss_store
from logger.logger import get_logger

# Import whylogs for monitoring FAISS retrieval and embedding metrics
import whylogs as why

# Import llm_metrics from langkit for LLM performance tracking.
from langkit import llm_metrics

logger = get_logger(__file__)

# -------------------------------------------------------------------
# 1) Prompt for LLM Re-ranking (for products)
# -------------------------------------------------------------------
template = """You are an intelligent assistant that reorders product IDs based on any constraints in the user query.

**Instructions:**
- You have a set of {doc_count} products, each with metadata: product_id, product_name, price, rating, etc.
- The user’s query might contain any constraint (e.g., a price limit, color requirement, brand name, rating threshold, etc.).
- First, identify which products match those constraints (the “most relevant” or direct matches) and place them **first**.
- Then list any remaining products afterward in any sensible order.
- **Return all product IDs** (you must not omit any), each on its own line, prefixed by "Product ID:".
- Provide **no extra commentary**, no disclaimers—just the lines "Product ID: XYZ".

Context:
{context}

Question: {question}

Helpful Answer:
"""

custom_rag_prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question", "doc_count"]
)

# -------------------------------------------------------------------
# 2) Define Pipeline State
# -------------------------------------------------------------------
class ProductState(TypedDict):
    question: str
    k: int                        # Number of documents to retrieve.
    context: List[Document]       # Holds up to k retrieved product documents.
    final_ids_output: str         # The final combined output after re-ranking.
    loop_step: int

# -------------------------------------------------------------------
# 3) Helper: Regex Parsing for Product IDs
# -------------------------------------------------------------------
def parse_product_ids_from_llm_text(llm_text: str) -> List[str]:
    pattern = r"Product ID:\s*(\S+)"
    matches = re.findall(pattern, llm_text)
    cleaned = [m.rstrip(",.") for m in matches]
    return cleaned

# -------------------------------------------------------------------
# 4) Helper: Compute Norm for Embeddings
# -------------------------------------------------------------------
def compute_norm(embedding):
    try:
        if embedding:
            return float(np.linalg.norm(np.array(embedding)))
    except Exception as e:
        logger.error("Error computing embedding norm", exc_info=True)
    return 0.0

# -------------------------------------------------------------------
# 5) Retrieval: Retrieve k Products and Log FAISS/Embedding Metrics
# -------------------------------------------------------------------
def retrieve_products(state: ProductState, faiss_store, embeddings) -> ProductState:
    try:
        k = state.get("k", 200)
        # Compute the query embedding using your Sentence-BERT model.
        query_embedding = embeddings.embed_text(state["question"])
        query_norm = compute_norm(query_embedding)
        
        retriever = faiss_store.as_retriever(search_kwargs={'k': k})
        docs = retriever.invoke(state["question"])
        logger.info(f"Retrieved {len(docs)} product documents with k={k}.")

        # Log similarity scores from each document (if available).
        similarity_scores = [doc.metadata.get("similarity_score", 0) for doc in docs]
        similarity_record = {"similarity_scores": similarity_scores}
        similarity_profile = why.log(similarity_record).profile()
        with open("logs/similarity_profile.json", "w") as f:
            json.dump(similarity_profile.to_summary_dict(), f)

        # Log embedding metrics: compute the norm for each document's embedding.
        doc_norms = [compute_norm(doc.metadata.get("embedding_vector", [])) for doc in docs]
        embedding_record = {"doc_embedding_norms": doc_norms, "query_embedding_norm": query_norm}
        embedding_profile = why.log(embedding_record).profile()
        with open("logs/embedding_profile.json", "w") as f:
            json.dump(embedding_profile.to_summary_dict(), f)

        # Also log basic retrieval details.
        retrieval_record = {
            "query": state["question"],
            "retrieved_count": len(docs)
        }
        retrieval_profile = why.log(retrieval_record).profile()
        with open("logs/retrieval_profile.json", "w") as f:
            json.dump(retrieval_profile.to_summary_dict(), f)

        return {"context": docs}
    except Exception as e:
        logger.error(f"Error retrieving {k} product docs: {e}", exc_info=True)
        return {"context": []}

# -------------------------------------------------------------------
# 6) Rerank Top 25 and Append Remainder; Track LLM Performance with llm_metrics
# -------------------------------------------------------------------
def rerank_and_append(state: ProductState, llm) -> ProductState:
    docs = state.get("context", [])
    if not docs:
        logger.warning("No product documents retrieved for re-ranking.")
        return {"final_ids_output": "No relevant product IDs found."}

    top25 = docs[:25]
    remainder = docs[25:]
    
    formatted_context = "\n".join(
        f"Product ID: {doc.metadata.get('product_id', 'Unknown')}, "
        f"Product Name: {doc.metadata.get('product_name', 'Unknown')}, "
        f"Price: {doc.metadata.get('price', 'Unknown')}, "
        f"Rating: {doc.metadata.get('rating', 'Unknown')}"
        for doc in top25
    )
    logger.info("Formatted context for top 25 products:\n" + formatted_context)
    
    prompt = custom_rag_prompt.format(
        doc_count=len(top25),
        context=formatted_context,
        question=state["question"]
    )

    # Log the prompt.
    prompt_record = {"prompt": prompt, "doc_count": len(top25)}
    prompt_profile = why.log(prompt_record).profile()
    with open("logs/prompt_profile.json", "w") as f:
        json.dump(prompt_profile.to_summary_dict(), f)
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        llm_text = response.content
        logger.info("LLM successfully re-ranked the top 25 product documents.")
        
        # Log the LLM response.
        response_record = {"llm_response": llm_text}
        response_profile = why.log(response_record).profile()
        with open("logs/llm_response_profile.json", "w") as f:
            json.dump(response_profile.to_summary_dict(), f)
        
        # Use llm_metrics (from langkit) to track the LLM's performance.
        # (Assuming llm_metrics takes the LLM output text and returns performance metrics.)
        llm_performance = llm_metrics(llm_text)
        with open("logs/llm_performance_metrics.json", "w") as f:
            json.dump(llm_performance, f)
        
        # Compute a basic hallucination metric by comparing expected vs. returned product IDs.
        original_ids = [str(doc.metadata.get("product_id", "Unknown")) for doc in top25]
        llm_ids = parse_product_ids_from_llm_text(llm_text)
        missing = [pid for pid in original_ids if pid not in set(llm_ids)]
        hallucination_metric = len(missing)
        logger.info(f"Hallucination metric (missing IDs count): {hallucination_metric}")
        
        metrics_record = {"hallucination_metric": hallucination_metric}
        metrics_profile = why.log(metrics_record).profile()
        with open("logs/metrics_profile.json", "w") as f:
            json.dump(metrics_profile.to_summary_dict(), f)
        
        # Append any missing IDs and remove duplicates.
        combined_top25 = llm_ids + missing
        seen = set()
        final_top25 = []
        for pid in combined_top25:
            if pid not in seen:
                seen.add(pid)
                final_top25.append(pid)
        
        remainder_ids = [f"Product ID: {doc.metadata.get('product_id', 'Unknown')}" for doc in remainder]
        final_output = "\n".join(f"Product ID: {pid}" for pid in final_top25)
        if remainder_ids:
            final_output += "\n" + "\n".join(remainder_ids)
        
        return {"final_ids_output": final_output}
    except Exception as e:
        logger.error("LLM re-ranking failed for top 25 product documents.", exc_info=True)
        return {"final_ids_output": "No relevant product IDs found."}

# -------------------------------------------------------------------
# 7) Build LangGraph Workflow (Updated to accept embeddings)
# -------------------------------------------------------------------
def build_product_graph(faiss_store, llm, embeddings):
    graph_builder = StateGraph(ProductState)
    graph_builder.add_node("retrieve", lambda s: retrieve_products(s, faiss_store, embeddings))
    graph_builder.add_node("rerank_and_append", lambda s: rerank_and_append(s, llm))
    graph_builder.add_edge(START, "retrieve")
    graph_builder.add_edge("retrieve", "rerank_and_append")
    compiled_graph = graph_builder.compile()
    logger.info("LangGraph product pipeline compiled successfully.")
    return compiled_graph

# -------------------------------------------------------------------
# 8) Main Execution (for local testing)
# -------------------------------------------------------------------
def main():
    try:
        # Ensure the logs directory exists.
        os.makedirs("logs", exist_ok=True)
        
        this_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(this_dir, "..", "config.yaml")
        config = load_config(config_path)
        logger.info(f"Configuration loaded from {config_path}")
        
        # Initialize the LLM.
        local_llm = "llama3.1:8b"
        llm = ChatOllama(model=local_llm, temperature=0)
        
        embeddings = initialize_embeddings(config["output_dir"])
        faiss_store = load_faiss_store(config["product_store_path"], embeddings)
        logger.info(f"FAISS vector store loaded from: {config['product_store_path']}")

        # Build the pipeline (now including embeddings for monitoring).
        graph = build_product_graph(faiss_store, llm, embeddings)

        while True:
            query = input("\nEnter your product search query (or type 'exit' to quit): ").strip()
            if query.lower() in ["exit", "quit"]:
                logger.info("Exiting the application.")
                break

            k_value_input = input("Enter number of documents to retrieve (k): ").strip()
            try:
                k_value = int(k_value_input)
            except ValueError:
                logger.warning("Invalid k value provided, defaulting to 200.")
                k_value = 200

            # Log user input.
            input_record = {"query": query, "k_value": k_value}
            input_profile = why.log(input_record).profile()
            with open("logs/input_profile.json", "w") as f:
                json.dump(input_profile.to_summary_dict(), f)

            initial_state: ProductState = {
                "question": query,
                "k": k_value,
                "context": [],
                "final_ids_output": "",
                "loop_step": 0,
            }

            final_state = graph.invoke(initial_state)
            final_ids = final_state.get("final_ids_output", "No response provided.")
            
            # Log the final output.
            final_record = {"final_ids_output": final_ids}
            final_profile = why.log(final_record).profile()
            with open("logs/final_profile.json", "w") as f:
                json.dump(final_profile.to_summary_dict(), f)

            print("\n=== Final Output ===")
            print(final_ids)
            print("====================\n")

    except KeyboardInterrupt:
        logger.info("Application interrupted by user. Exiting gracefully.")

if __name__ == "__main__":
    main()
