import mlflow
from sentence_transformers import SentenceTransformer
from logger.logger import get_logger
import torch
logger = get_logger(__file__)
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

# Example usage:
if __name__ == "__main__":
    model_uri = "models:/jsearch_model/12"  # or "models:/jsearch_model/staging"
    model = initialize_mlflow_model(model_uri)
    embeddings = model.encode(["Your sample text here"])
    print(embeddings)