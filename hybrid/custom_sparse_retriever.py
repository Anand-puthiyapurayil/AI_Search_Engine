from typing import List
from langchain_core.documents import Document
from langchain_core.runnables.base import Runnable  # Adjust import if necessary
from utils.utils import load_config 
from elasticsearch import Elasticsearch
import os



def retrieve_documents_with_metadata(es_client, index_name, query, size=10) -> List[Document]:
    response = es_client.search(
        index=index_name,
        body={
            "query": {"match": {"content": query}},
            "size": size
        }
    )
    return [
        Document(
            page_content=hit["_source"].get("content", ""),
            metadata={k: v for k, v in hit["_source"].items() if k != "content"}
        )
        for hit in response["hits"]["hits"]
    ]

class CustomSparseRetriever(Runnable):
    def __init__(self, es_client, index_name, size=10):
        self.es_client = es_client
        self.index_name = index_name
        self.size = size

    def invoke(self, query: str, run_manager=None, **kwargs) -> List[Document]:
        return retrieve_documents_with_metadata(self.es_client, self.index_name, query, size=self.size)

if __name__ == '__main__':
    # Load configuration
    this_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(this_dir, "..", "config.yaml")
    config = load_config(config_path)
    # Set up Elasticsearch client and BM25 index details
    es_url = config["elasticsearch_url"]
    bm25_index = config.get("product_index_name", "products")
    es_client = Elasticsearch(es_url)

    # Test the retrieval function with a sample query
    query = "i need aluminum"  # Replace with your actual search term
    documents = retrieve_documents_with_metadata(es_client, bm25_index, query,size=50)
    
    # Print out the results
    for doc in documents:
        print("Content:", doc.page_content)
        print("Metadata:", doc.metadata)
        print("-" * 40)