import numpy as np
from utils.utils import load_config, initialize_embeddings
import os
this_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(this_dir, "..", "config.yaml")
config = load_config(config_path)

embeddings = initialize_embeddings(config["output_dir"])
def normalize_vector(vec):
    norm = np.linalg.norm(vec)
    return vec if norm == 0 else vec / norm

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# --- For the Query ---
query = "show me an aluminum from canada california and los angeles"
query_vector = embeddings.embed_query(query)
normalized_query_vector = normalize_vector(np.array(query_vector))
print("Query Vector:")
print(query_vector)
print("Normalized Query Vector:")
print(normalized_query_vector)

# --- For a Specific Product (P000001) ---
product_text = (
    "Product ID: P000001\n"
    "Product Name: Aluminum 907\n"
    "Description: High-quality aluminum for industrial use.\n"
    "Supplier: ElectroTech\n"
    "Location: Los Angeles, California, Canada\n"
    "Store: Warehouse A\n"
    "Category Hierarchy: Metals > Aluminum > Aluminum - Type 10\n"
    "Business Source: B2C\n"
    "Variant: V10\n"
)

# Since the embeddings object uses embed_documents, pass a list and get the first element:
product_vector_list = embeddings.embed_documents([product_text])
product_vector = product_vector_list[0]

normalized_product_vector = normalize_vector(np.array(product_vector))
print("Product P000001 Vector:")
print(product_vector)
print("Normalized Product P000001 Vector:")
print(normalized_product_vector)

# --- Compare Vectors via Cosine Similarity ---
similarity = cosine_similarity(query_vector, product_vector)
print("Cosine Similarity between query and product P000001:")
print(similarity)
