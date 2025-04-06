import os
import pandas as pd
from utils.utils import load_config, load_dataframe, initialize_embeddings, add_documents_to_store, build_faiss_vector_store
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from logger.logger import get_logger

# Initialize logger
logger = get_logger(__file__)

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
            # Improved full text embedding with explicit keywords
            full_text = (
                f"product Name: {row['product_name']}\n"
                f"supplier: {row['supplier_name'] if not pd.isna(row['supplier_name']) else 'N/A'}\n"
                f"city: {row['city']}\n"
                f"state: {row['state']}\n"
                f"country: {row['country']}\n"
                f"category: {row['category']}\n"
                f"subcategory: {row['subcategory']}\n"
                f"sub-subcategory: {row['subsubcategory'] if not pd.isna(row['subsubcategory']) else 'N/A'}\n"
                f"variant: {row['variation'] if not pd.isna(row['variation']) else 'N/A'}"
             )
            

            # Splitting text into chunks (rarely needed for short text like products)
            chunks = text_splitter.split_text(full_text)

            metadata = {
                "product_id": row['product_id'],
                "product_name": row['product_name'],
                "city": row['city'],
                "state": row['state'],
                "country": row['country'],
                "variant": row['variation'] if not pd.isna(row['variation']) else 'N/A',
                "category": row['category'],
                "subcategory": row['subcategory'],
                "subsubcategory": row['subsubcategory'] if not pd.isna(row['subsubcategory']) else 'N/A',
                "Supplier": row['supplier_name']
            }

            for chunk in chunks:
                documents.append(Document(page_content=chunk, metadata=metadata))

        logger.info(f"Created {len(documents)} documents from DataFrame.")

        return documents
    except Exception as e:
        logger.error(f"Failed to create documents: {e}", exc_info=True)
        raise e


# ----------------------------------------------------------
# 4) Main Pipeline Execution
# ----------------------------------------------------------
def main():
    try:
        # Set up directory paths and configuration file
        this_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(this_dir, "..", "config.yaml")
        config = load_config(config_path)

        # Load DataFrame from CSV
        df = load_dataframe(config["product_data"])

        # Create chunked documents with metadata using the improved splitter strategy
        documents = create_documents(
            df,
            chunk_size=config.get("chunk_size", 1000),
            chunk_overlap=config.get("chunk_overlap", 100)
        )

        # Initialize the embedding function
        embeddings = initialize_embeddings(config["output_dir"])

        # Build the FAISS vector store (custom approach)
        vector_store = build_faiss_vector_store(embeddings)

        # Add documents to the vector store in batches
        add_documents_to_store(documents, vector_store)
        
        num_vectors = vector_store.index.ntotal
        logger.info(f"FAISS index contains {num_vectors} vectors.")
        print(f"FAISS index contains {num_vectors} vectors.")

        # Save the FAISS vector store locally
        os.makedirs(config["product_store_path"], exist_ok=True)
        vector_store.save_local(config["product_store_path"])
        logger.info(f"FAISS vector store saved to {config['product_store_path']}")

        logger.info("Pipeline executed successfully.")
    except Exception as e:
        logger.error(f"An error occurred in the main function: {e}", exc_info=True)
        raise e

# ----------------------------------------------------------
# Entry Point
# ----------------------------------------------------------
if __name__ == "__main__":
    main()
