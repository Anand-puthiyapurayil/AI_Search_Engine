import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
import yaml
import os

# Import your custom logger
from logger.logger import get_logger

# Create a logger for this script
logger = get_logger(__file__)

###########################
# 1) LOAD CONFIG
###########################
def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

###########################
# 2) AUTOMATION FUNCTION
###########################
def automate_staging_to_production(model_registry_name):
    """
    Automates the promotion of the latest model version to 'Production' using tags.
    Archives all other production versions by updating their tags.

    Args:
        model_registry_name (str): Name of the model in the MLflow Model Registry.
    """
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("http://192.168.1.227:5000")  # Ensure this matches your MLflow server
    logger.info("MLflow Tracking URI set to http://192.168.1.227:5000")

    client = MlflowClient()

    try:
        # Fetch all model versions for the specified model
        all_versions = client.search_model_versions(f"name='{model_registry_name}'")
        staging_versions = []

        logger.info(f"Total versions found for model '{model_registry_name}': {len(all_versions)}")
        for version in all_versions:
            version_num = version.version
            model_version_details = client.get_model_version(name=model_registry_name, version=version_num)
            
            # Retrieve the 'stage' tag if it exists
            stage_tag = model_version_details.tags.get("stage", "").lower()
            logger.info(f"Version {version_num}: stage = {stage_tag}")
            
            if stage_tag == "staging":
                staging_versions.append(model_version_details)

        if not staging_versions:
            logger.warning(f"No 'staging' version found for model '{model_registry_name}'. Exiting.")
            return

        # Select the latest staging version based on version number
        latest_staging_version = max(staging_versions, key=lambda v: int(v.version))
        version_num = latest_staging_version.version
        logger.info(f"Promoting version {version_num} to 'Production'...")

        # Use tags to mark the latest version as 'Production'
        client.set_model_version_tag(
            name=model_registry_name,
            version=version_num,
            key="stage",
            value="Production"
        )
        
        # Archive all other 'Production' versions
        for version in all_versions:
            if version.version != version_num:
                client.set_model_version_tag(
                    name=model_registry_name,
                    version=version.version,
                    key="stage",
                    value="Archived"
                )
        logger.info(f"Version {version_num} promoted to 'Production'. Archived other versions.")

    except MlflowException as e:
        logger.error(f"An MLflow exception occurred: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")

###########################
# 3) MAIN EXECUTION
###########################
if __name__ == "__main__":
    # Determine the path to config.yaml relative to this script
    this_dir = os.path.dirname(__file__)
    config_path = os.path.join(this_dir, "..", "config.yaml")  # Adjust path as needed

    # Load configuration
    cfg = load_config(config_path)
    logger.info("Loaded config.")

    # Extract the model registry name from config
    model_registry_name = cfg.get("model_registry_name")
    if not model_registry_name:
        logger.error("Model registry name not found in config.yaml. Please update the configuration.")
        exit(1)

    logger.info(f"Starting the automation process for model: {model_registry_name}")
    automate_staging_to_production(model_registry_name)
    logger.info("Process completed.")
