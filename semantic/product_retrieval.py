import os
import re
from typing import List, TypedDict

# LangGraph
from langgraph.graph import START, StateGraph

# LangChain / LLM / Documents
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

# Utilities (Replace with your actual paths, or remove if not needed)
from utils.utils import (
    load_config,
    initialize_embeddings,
    load_faiss_store
)
from logger.logger import get_logger

logger = get_logger(__name__)

# -------------------------------------------------------------------
# 1) Define the ProductState
# -------------------------------------------------------------------
class ProductState(TypedDict):
    question: str
    k: int
    context: List[Document]     # Docs retrieved from FAISS
    faiss_metadata: str         # Assembled doc metadata
    llm_text: str               # Final LLM output text
    loop_step: int

# -------------------------------------------------------------------
# 2) Node: Retrieve Products from FAISS
# -------------------------------------------------------------------
def retrieve_products(state: ProductState, faiss_store) -> ProductState:
    """
    Retrieves up to k product documents from the FAISS store based on the user's query.
    """
    try:
        k = state.get("k", 200)
        docs = faiss_store.similarity_search(state["question"], k=k)
        logger.info(f"Retrieved {len(docs)} product documents with k={k}.")
        return {"context": docs}
    except Exception as e:
        logger.error(f"Error retrieving product docs: {e}", exc_info=True)
        return {"context": []}

# -------------------------------------------------------------------
# 3) Node: Gather FAISS Metadata
# -------------------------------------------------------------------
def gather_faiss_metadata(state: ProductState) -> ProductState:
    """
    Reads all docs in `state["context"]` and assembles metadata lines into `faiss_metadata`.
    """
    docs = state.get("context", [])
    if not docs:
        logger.warning("No documents found to gather metadata.")
        return {"faiss_metadata": "No documents retrieved; no metadata."}
    
    lines = []
    for doc in docs:
        items = [f"{key}: {val}" for key, val in doc.metadata.items()]
        lines.append(", ".join(items))
    metadata_str = "\n".join(lines)
    
    logger.info("FAISS metadata successfully gathered.")
    return {"faiss_metadata": metadata_str}

# -------------------------------------------------------------------
# 4) Node: Call LLM (Streaming)
# -------------------------------------------------------------------
# We'll define a prompt for the LLM.
custom_recommendation_prompt = PromptTemplate(
    template="""You are an intelligent assistant that summarizes the product(s) based on the user's query from the available products.

**Instructions:**
- Generate a concise product(s) info for all the available product(s) in a paragraph, naturally integrating specific product details such as product name, location, and variant.
- Suggest related products that the user might find interesting within the context.
- Do not introduce or suggest any products that are not mentioned in the context
- If the user's query is vague or general, gently prompt the user by suggesting specific product names or categories to explore further.
- Encourage user interaction by inviting them to refine your query or ask follow-up questions for more tailored information.

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
    Builds a prompt from up to 25 docs, calls the LLM, and stores the final text in `llm_text`.
    The actual live streaming is handled in graph.stream(...) at runtime.
    """
    docs = state.get("context", [])
    if not docs:
        logger.warning("No documents found for LLM recommendation.")
        return {"llm_text": "No documents for LLM."}
    
    # Take top 25 for context
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
# 5) Build Pipeline A: FAISS Only
# -------------------------------------------------------------------
def build_faiss_pipeline(faiss_store):
    """
    Pipeline A: 
      START -> retrieve_products -> gather_faiss_metadata (end)
    """
    graph_builder = StateGraph(ProductState)
    
    graph_builder.add_node("retrieve_products", lambda s: retrieve_products(s, faiss_store))
    graph_builder.add_node("gather_faiss_metadata", gather_faiss_metadata)
    
    graph_builder.add_edge(START, "retrieve_products")
    graph_builder.add_edge("retrieve_products", "gather_faiss_metadata")
    
    compiled_graph = graph_builder.compile()
    logger.info("FAISS-only pipeline compiled.")
    return compiled_graph

# -------------------------------------------------------------------
# 6) Build Pipeline B: LLM Only (1 node)
# -------------------------------------------------------------------
def build_llm_pipeline(llm):
    """
    Pipeline B:
      START -> call_llm_recommendation (end)
    """
    graph_builder = StateGraph(ProductState)
    
    graph_builder.add_node("call_llm_recommendation", lambda s: call_llm_recommendation(s, llm))
    
    graph_builder.add_edge(START, "call_llm_recommendation")
    
    compiled_graph = graph_builder.compile()
    logger.info("LLM-only pipeline compiled.")
    return compiled_graph

# -------------------------------------------------------------------
# 7) Main Execution: 
#    - 1) Invoke the FAISS pipeline for final result
#    - 2) Stream the LLM pipeline for tokens
# -------------------------------------------------------------------
def main():
    try:
        # 1) Load config, LLM, FAISS
        this_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(this_dir, "..", "config.yaml")
        config = load_config(config_path)
        logger.info(f"Configuration loaded from {config_path}")
        
        llm = ChatOllama(
            model="llama3.1:8b",
            stream=True,   # Important for streaming tokens
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
        logger.info(f"FAISS store loaded from: {config['product_store_path']}")
        
        # 2) Build two pipelines
        faiss_pipeline = build_faiss_pipeline(faiss_store)
        llm_pipeline = build_llm_pipeline(llm)
        
        # 3) Interactive Loop
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
            
            # ----------------------------------------------------------
            # A) First pipeline: FAISS retrieval + metadata (invoke)
            # ----------------------------------------------------------
            faiss_final = faiss_pipeline.invoke(initial_state)
            
            # Print the final FAISS metadata
            print("\n=== FAISS METADATA ===")
            print(faiss_final.get("faiss_metadata", "No metadata found."))
            print("======================\n")
            
            # ----------------------------------------------------------
            # B) Second pipeline: LLM streaming
            #     We pass `faiss_final` so it includes docs, if needed
            # ----------------------------------------------------------
            final_llm_state = None
            print("=== STREAMING LLM TOKENS ===")
            
            for msg, metadata in llm_pipeline.stream(
                faiss_final,
                stream_mode="messages"
            ):
                # The only node is "call_llm_recommendation"
                if metadata["langgraph_node"] == "call_llm_recommendation":
                    print(msg.content, end="", flush=True)
                
                if metadata.get("langgraph_phase") == "end":
                    final_llm_state = metadata["final_state"]
            
            print("\n=== END OF LLM STREAMING ===\n")
            
            
    except KeyboardInterrupt:
        logger.info("User interrupted. Exiting gracefully.")

if __name__ == "__main__":
    main()
