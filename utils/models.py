import os
import torch
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
from sentence_transformers import SentenceTransformer, CrossEncoder

from utils.utils import load_config
from logger.logger import get_logger

logger = get_logger(__file__)

def save_models_without_finetuning(
    base_biencoder_model: str,
    base_cross_encoder_model: str,
    output_dir_sbert: str,
    output_dir_cross_encoder: str,
    experiment_name: str,
    sbert_registry_name: str,
    cross_encoder_registry_name: str,
    initial_stage: str = "staging"
):
    """
    Loads a base SentenceTransformer (bi-encoder) model and a CrossEncoder model,
    logs them to MLflow WITHOUT fine-tuning, and also saves each locally.
    """

    # Configure MLflow (update the tracking URI as needed)
    mlflow.set_tracking_uri("http://192.168.1.227:5000")
    client = MlflowClient()
    mlflow.set_experiment(experiment_name)

    # Determine the device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # -----------------------------------------------------------------
    # 1) Load and Save Bi-Encoder (SentenceTransformer)
    # -----------------------------------------------------------------
    sbert_model = SentenceTransformer(base_biencoder_model, device=device)

    # Save the SBERT model locally
    os.makedirs(output_dir_sbert, exist_ok=True)
    sbert_model.save(output_dir_sbert)
    logger.info(f"Bi-encoder model saved locally at: {output_dir_sbert}")

    # -----------------------------------------------------------------
    # 2) Load and Save Cross-Encoder
    # -----------------------------------------------------------------
    cross_encoder = CrossEncoder(base_cross_encoder_model, device=device)

    # Save the CrossEncoder model locally
    os.makedirs(output_dir_cross_encoder, exist_ok=True)
    cross_encoder.save(output_dir_cross_encoder)
    logger.info(f"Cross-encoder model saved locally at: {output_dir_cross_encoder}")

    # -----------------------------------------------------------------
    # 3) Log Both Models to MLflow
    # -----------------------------------------------------------------
    with mlflow.start_run() as run:
        run_id = run.info.run_id

        # Log minimal params
        mlflow.log_param("base_biencoder_model", base_biencoder_model)
        mlflow.log_param("base_cross_encoder_model", base_cross_encoder_model)

        # A) Log the SBERT model with mlflow.sentence_transformers
        logged_sbert = mlflow.sentence_transformers.log_model(
            model=sbert_model,
            artifact_path="sbert_model"
        )
        logger.info("SBERT model logged via mlflow.sentence_transformers.log_model")

        # B) Log the CrossEncoder model as generic artifacts
        #    CrossEncoder is NOT a SentenceTransformer, so we use log_artifacts
        mlflow.log_artifacts(
            local_dir=output_dir_cross_encoder,
            artifact_path="cross_encoder_artifacts"
        )
        logger.info("CrossEncoder model logged as artifacts under 'cross_encoder_artifacts'")

        # -----------------------------------------------------------------
        # 4) Register Each Model in MLflow Model Registry
        # -----------------------------------------------------------------
        # i) SBERT in registry
        try:
            client.create_registered_model(sbert_registry_name)
        except MlflowException as e:
            if "RESOURCE_ALREADY_EXISTS" not in str(e):
                raise e

        sbert_model_version = client.create_model_version(
            name=sbert_registry_name,
            source=logged_sbert.model_uri,  # from log_model
            run_id=run_id
        )
        if sbert_model_version:
            client.set_model_version_tag(
                name=sbert_registry_name,
                version=sbert_model_version.version,
                key="stage",
                value=initial_stage
            )
        logger.info(f"SBERT registered as: {sbert_registry_name}, version: {sbert_model_version.version}")

        # ii) CrossEncoder in registry
        # We'll point the 'source' to the same run's artifact URI for cross_encoder_artifacts
        crossencoder_source = f"{mlflow.get_artifact_uri()}/cross_encoder_artifacts"
        try:
            client.create_registered_model(cross_encoder_registry_name)
        except MlflowException as e:
            if "RESOURCE_ALREADY_EXISTS" not in str(e):
                raise e

        cross_encoder_model_version = client.create_model_version(
            name=cross_encoder_registry_name,
            source=crossencoder_source,
            run_id=run_id
        )
        if cross_encoder_model_version:
            client.set_model_version_tag(
                name=cross_encoder_registry_name,
                version=cross_encoder_model_version.version,
                key="stage",
                value=initial_stage
            )
        logger.info(f"CrossEncoder registered as: {cross_encoder_registry_name}, version: {cross_encoder_model_version.version}")

    logger.info("Bi-encoder & Cross-encoder saved locally and logged to MLflow without fine-tuning.")


def main():
    try:
        # Load config (which should contain keys for both models and their output dirs)
        this_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(this_dir, "..", "config.yaml")
        config = load_config(config_path)

        # Example: we expect these keys in config:
        #   "base_model" -> e.g. "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
        #   "base_cross_encoder" -> e.g. "cross-encoder/ms-marco-MiniLM-L-12-v2"
        #   "output_dir" -> path like "output/sbert"
        #   "output_dir_cross_encoder" -> path like "output/crossencoder"
        #   "experiment_name" -> e.g. "MyExperiment"
        #   "model_registry_name" -> e.g. "MySBERTRegistry"
        #   "cross_encoder_registry_name" -> e.g. "MyCrossEncoderRegistry"
        #   "initial_stage" -> e.g. "staging"

        save_models_without_finetuning(
            base_biencoder_model=config["base_model"],
            base_cross_encoder_model=config["base_cross_encoder"],
            output_dir_sbert=config["output_dir"],
            output_dir_cross_encoder=config["output_dir_cross_encoder"],
            experiment_name=config["experiment_name"],
            sbert_registry_name=config["model_registry_name"],
            cross_encoder_registry_name=config["cross_encoder_registry_name"],
            initial_stage=config.get("initial_stage", "staging"),
        )
        logger.info("Models saved without fine-tuning successfully.")
    except Exception as e:
        logger.error(f"Error in main: {e}", exc_info=True)
        raise e

if __name__ == "__main__":
    main()
