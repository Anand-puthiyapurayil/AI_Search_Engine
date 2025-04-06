import os
import pandas as pd
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk, BulkIndexError
from utils.utils import load_config, load_dataframe, initialize_embeddings, add_documents_to_store, build_faiss_vector_store
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from logger.logger import get_logger

logger = get_logger(__file__)

# Helper: safely convert a value to lowercase (handles NaN)
def safe_lower(value):
    if pd.isna(value):
        return ''
    return str(value).lower().strip()

# Helper: safely convert a value to a string, replacing NaN with "N/A"
def safe_str(value):
    if pd.isna(value):
        return "N/A"
    return str(value)

# ----------------------------------------------------------
# 1) Create Chunked Documents with Metadata
# ----------------------------------------------------------
def create_documents(df: pd.DataFrame, chunk_size: int, chunk_overlap: int):
    try:
        documents = []
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n"]
        )

        for _, row in df.iterrows():
            # Build full text using safe_str to replace NaN with "N/A"
            full_text = (
                f"product Name: {safe_str(row['product_name'])}\n"
                f"supplier: {safe_str(row['supplier_name'])}\n"
                f"city: {safe_str(row['city'])}\n"
                f"state: {safe_str(row['state'])}\n"
                f"country: {safe_str(row['country'])}\n"
                f"category: {safe_str(row['category'])}\n"
                f"subcategory: {safe_str(row['subcategory'])}\n"
                f"sub-subcategory: {safe_str(row['subsubcategory'])}\n"
                f"variant: {safe_str(row['variation'])}"
            )
            
            # Splitting text into chunks (rarely needed for short text like products)
            chunks = text_splitter.split_text(full_text)

            metadata = {
                "product_id": row['product_id'],
                "product_name": safe_str(row['product_name']),
                "city": safe_str(row['city']),
                "state": safe_str(row['state']),
                "country": safe_str(row['country']),
                "variant": safe_str(row['variation']),
                "category": safe_str(row['category']),
                "subcategory": safe_str(row['subcategory']),
                "subsubcategory": safe_str(row['subsubcategory']),
                "Supplier": safe_str(row['supplier_name'])
            }

            for chunk in chunks:
                documents.append(Document(page_content=chunk, metadata=metadata))

        logger.info(f"Created {len(documents)} documents from DataFrame.")
        return documents

    except Exception as e:
        logger.error(f"Failed to create documents: {e}", exc_info=True)
        raise e

# ----------------------------------------------------------
# 2) Convert Document to Structured Index Document for BM25
# ----------------------------------------------------------
def create_index_document(doc: Document, index_name: str) -> dict:
    """
    Convert a Document into a dictionary for Elasticsearch's bulk API.
    
    For Elasticsearch indexing, modify the document's page content to remove the lines for
    'category', 'subcategory', and 'sub-subcategory'. The metadata remains unchanged.
    """
    # Filter out lines in the page content that start with "category:", "subcategory:" or "sub-subcategory:"
    lines = doc.page_content.split("\n")
    filtered_lines = [
        line for line in lines
        if not (line.lower().startswith("category:") or 
                line.lower().startswith("subcategory:") or 
                line.lower().startswith("sub-subcategory:"))
    ]
    modified_content = "\n".join(filtered_lines)
    
    # Keep metadata unchanged (they already have safe string values).
    metadata = doc.metadata.copy()
    
    return {
        "_op_type": "index",
        "_index": index_name,
        "content": modified_content,
        **metadata
    }

# ----------------------------------------------------------
# 3) Main Pipeline Execution
# ----------------------------------------------------------
def main():
    try:
        # Load configuration and data
        this_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(this_dir, "..", "config.yaml")
        config = load_config(config_path)
        df = load_dataframe(config["product_data"])

        # Create documents from DataFrame
        documents = create_documents(
            df,
            chunk_size=config.get("chunk_size", 1000),
            chunk_overlap=config.get("chunk_overlap", 100)
        )

        # Optional: Initialize dense retrieval (FAISS) if needed
        embeddings = initialize_embeddings(config["output_dir"])
        vector_store = build_faiss_vector_store(embeddings)
        add_documents_to_store(documents, vector_store)

        num_vectors = vector_store.index.ntotal
        logger.info(f"FAISS index contains {num_vectors} vectors.")
        print(f"FAISS index contains {num_vectors} vectors.")

        os.makedirs(config["product_store_path"], exist_ok=True)
        vector_store.save_local(config["product_store_path"])
        logger.info(f"FAISS vector store saved to {config['product_store_path']}")

        # Set up Elasticsearch client and index documents for sparse retrieval
        es_url = config["elasticsearch_url"]
        product_index = config.get("product_index_name", "products")
        es_client = Elasticsearch(es_url)

        # Delete existing index if it exists
        if es_client.indices.exists(index=product_index):
            es_client.indices.delete(index=product_index)
            logger.info(f"Deleted existing index: {product_index}")

        index_documents_list = [create_index_document(doc, product_index) for doc in documents]

        try:
            bulk(es_client, index_documents_list)
        except BulkIndexError as e:
            logger.error(f"Bulk indexing error: {e.errors}")
            raise e

        logger.info("Elasticsearch BM25 index created and all documents indexed.")
        es_client.indices.refresh(index=product_index)

    except Exception as e:
        logger.error(f"An error occurred in the main function: {e}", exc_info=True)
        raise e

if __name__ == "__main__":
    main()
