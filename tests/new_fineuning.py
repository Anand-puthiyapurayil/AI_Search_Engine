import os
import random
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
import yaml

import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses

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
# 2) DATA & TRIPLET CREATION
###########################
###########################
# 2) DATA & TRIPLET CREATION
###########################
from collections import defaultdict

def create_triplets_for_products(data_list):
    """
    Generate all possible unique triplets (anchor, positive, negative) based on categories and subcategories.
    
    Args:
        data_list (list): List of product dictionaries.
        
    Returns:
        list: List of InputExample triplets.
    """
    triplets = []
    
    # Precompute category and subcategory mappings for efficient lookup
    category_map = defaultdict(list)
    subcategory_map = defaultdict(list)
    
    for item in data_list:
        category_map[item['category']].append(item)
        subcategory_map[item['subcategory']].append(item)
    
    def build_text(item):
        return f"{item['product_name']}. {item['description']} (Attrs: {item['attributes']})"
    
    # Iterate through each product as the anchor
    for anchor in data_list:
        # Find all positives: Same subcategory excluding the anchor itself
        positives = [p for p in subcategory_map[anchor['subcategory']] if p != anchor]
        
        # If no positives in the same subcategory, use the same category
        if not positives:
            positives = [p for p in category_map[anchor['category']] if p != anchor]
        
        # Skip if no valid positives are found
        if not positives:
            logger.warning(f"No positives found for anchor: {anchor['product_name']}. Skipping.")
            continue
        
        # Find all negatives: Different category
        negatives = [n for n in data_list if n['category'] != anchor['category']]
        
        # Skip if no valid negatives are found
        if not negatives:
            logger.warning(f"No negatives found for anchor: {anchor['product_name']}. Skipping.")
            continue
        
        # Create triplets for each positive and negative pair
        for positive in positives:
            for negative in negatives:
                triplet = InputExample(
                    texts=[
                        build_text(anchor),
                        build_text(positive),
                        build_text(negative)
                    ]
                )
                triplets.append(triplet)
    
    logger.info(f"Total triplets generated: {len(triplets)}")
    return triplets

###########################
# 3) FINE-TUNE + REGISTRY WITH TAGS
###########################
def fine_tune_sbert(
    triplets,
    base_model,
    output_dir,
    experiment_name,
    epochs,
    batch_size,
    model_registry_name,
    initial_stage="staging"  # Consistent tag value
):
    """
    Fine-tunes SBERT, logs to an MLflow server at http://127.0.0.1:5000,
    registers the new model version under `model_registry_name`,
    and assigns a stage (by default, "staging") to the newly created version.

    Make sure the MLflow server is running with a DB backend.
    """
    # 1) Connect to MLflow server
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    logger.info("MLflow Tracking URI => http://127.0.0.1:5000")

    client = MlflowClient()

    # 2) Create / set experiment
    mlflow.set_experiment(experiment_name)
    logger.info(f"Using experiment => {experiment_name}")

    # 3) Start an MLflow run
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info(f"Started MLflow run => ID: {run_id}")

        # 4) Load SBERT
        logger.info(f"Loading base model => {base_model}")
        model = SentenceTransformer(base_model)

        # 5) Prep DataLoader & Loss
        train_loader = DataLoader(triplets, shuffle=True, batch_size=batch_size)
        train_loss = losses.TripletLoss(model)
        warmup_steps = int(0.1 * len(train_loader) * epochs)

        # Log parameters
        mlflow.log_param("base_model", base_model)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)

        logger.info(f"Training on {len(triplets)} triplets; epochs={epochs}, batch_size={batch_size}")

        # 6) Fine-tune
        logger.info("Fine-tuning started...")
        model.fit(
            train_objectives=[(train_loader, train_loss)],
            epochs=epochs,
            warmup_steps=warmup_steps,
            output_path=output_dir
        )
        logger.info("Fine-tuning complete.")

        # 7) Log metrics & artifacts
        mlflow.log_metric("dummy_metric", 0.99)  # Replace with actual metrics if available
        logger.info("dummy_metric => 0.99")

        # 8) Log the model using mlflow-sentence-transformers without signature or input_example
        logged_model = mlflow.sentence_transformers.log_model(
            model=model,
            artifact_path="sbert_model"
            # No signature or input_example provided
        )
        logger.info(f"Model logged with artifact path 'sbert_model'.")

        # 9) Ensure the registered model exists
        try:
            client.create_registered_model(model_registry_name)
            logger.info(f"Registered model '{model_registry_name}' created.")
        except MlflowException as e:
            if "RESOURCE_ALREADY_EXISTS" in str(e):
                logger.info(f"Registered model '{model_registry_name}' already exists.")
            else:
                logger.error(f"Error creating registered model: {e}")
                raise e

        # 10) Register the model version under the registered model name
        logger.info(f"Registering model version for '{model_registry_name}'.")
        model_version = client.create_model_version(
            name=model_registry_name,            # Specify the registered model name here
            source=logged_model.model_uri,
            run_id=run_id
        )
        logger.info(f"Model version {model_version.version} registered under '{model_registry_name}'.")

        # 11) Set stage tag for the newly created version
        if model_version is not None:
            logger.info(f"Setting tag 'stage'='{initial_stage}' on version={model_version.version}.")
            client.set_model_version_tag(
                name=model_registry_name,
                version=model_version.version,
                key="stage",
                value=initial_stage
            )
            logger.info(f"Tag 'stage' set to '{initial_stage}' for version={model_version.version}.")
        else:
            logger.warning("Could not find the newly created model version. Tag not set.")

    logger.info(f"MLflow run {run_id} finished successfully.")

###########################
# 4) MAIN
###########################
if __name__ == "__main__":
    this_dir = os.path.dirname(__file__)
    config_path = os.path.join(this_dir, "..", "config.yaml")  # Adjusted path as needed

    cfg = load_config(config_path)
    print("Loaded config:", cfg)

    # Real product data
    data_list = [
        {
            "product_name": "Cordless Drill",
            "description": "Lightweight cordless drill with long battery life and powerful motor.",
            "category": "Tools",
            "subcategory": "Power Tools",
            "subsubcategory": "Drills",
            "attributes": {"Power": "18V", "Battery Life": "3 hours", "Weight": "1.5kg"},
            "features": ["Cordless", "Ergonomic Design", "Multiple Speed Settings"]
        },
        {
            "product_name": "Impact Wrench",
            "description": "Compact and powerful wrench for automotive and industrial applications.",
            "category": "Tools",
            "subcategory": "Power Tools",
            "subsubcategory": "Wrenches",
            "attributes": {"Torque": "300 Nm", "Power Source": "Corded", "Weight": "2kg"},
            "features": ["Compact Design", "High Torque", "Durable Build"]
        },
        {
            "product_name": "Arc Welding Machine",
            "description": "Portable arc welding machine suitable for heavy-duty industrial use.",
            "category": "Machinery",
            "subcategory": "Welding Equipment",
            "subsubcategory": "Arc Welders",
            "attributes": {"Voltage": "220V", "Amperage": "200A", "Weight": "7kg"},
            "features": ["Portable", "High Efficiency", "Thermal Overload Protection"]
        },
        {
            "product_name": "Conveyor Belt",
            "description": "Durable conveyor belt for factory automation and material handling.",
            "category": "Machinery",
            "subcategory": "Material Handling",
            "subsubcategory": "Conveyor Systems",
            "attributes": {"Length": "50m", "Width": "1m", "Material": "Rubber"},
            "features": ["Heat Resistant", "Heavy Load Capacity", "Easy Installation"]
        },
        {
            "product_name": "Hydraulic Jack",
            "description": "Heavy-duty hydraulic jack for lifting vehicles and machinery.",
            "category": "Automotive",
            "subcategory": "Lifting Equipment",
            "subsubcategory": "Hydraulic Jacks",
            "attributes": {"Lifting Capacity": "5 tons", "Material": "Steel", "Weight": "8kg"},
            "features": ["Heavy-Duty", "Compact", "Safety Valve"]
        },
        {
            "product_name": "Table Saw",
            "description": "High-precision table saw for woodworking and professional applications.",
            "category": "Tools",
            "subcategory": "Power Tools",
            "subsubcategory": "Saws",
            "attributes": {"Blade Size": "10 inches", "Power": "15A", "Weight": "30kg"},
            "features": ["High Precision", "Safety Guard", "Durable"]
        },
        {
            "product_name": "Forklift",
            "description": "Electric forklift for efficient material handling in warehouses.",
            "category": "Machinery",
            "subcategory": "Material Handling",
            "subsubcategory": "Forklifts",
            "attributes": {"Capacity": "3 tons", "Battery Life": "8 hours", "Height": "3m"},
            "features": ["Electric Powered", "Low Maintenance", "Eco-Friendly"]
        },
        {
            "product_name": "Air Compressor",
            "description": "Portable air compressor for pneumatic tools and inflating tires.",
            "category": "Tools",
            "subcategory": "Air Tools",
            "subsubcategory": "Compressors",
            "attributes": {"Tank Size": "20L", "Power": "1.5HP", "Weight": "12kg"},
            "features": ["Portable", "Quiet Operation", "Fast Filling"]
        },
        {
            "product_name": "Pressure Washer",
            "description": "High-pressure washer for cleaning vehicles and industrial equipment.",
            "category": "Tools",
            "subcategory": "Cleaning Equipment",
            "subsubcategory": "Pressure Washers",
            "attributes": {"Pressure": "150 bar", "Flow Rate": "8L/min", "Power": "2.5HP"},
            "features": ["High Pressure", "Energy Efficient", "Durable Hose"]
        },
        {
            "product_name": "Ladder",
            "description": "Extendable aluminum ladder for household and industrial use.",
            "category": "Tools",
            "subcategory": "Ladders",
            "subsubcategory": "Extendable Ladders",
            "attributes": {"Height": "5m", "Material": "Aluminum", "Weight": "5kg"},
            "features": ["Lightweight", "Rust-Resistant", "Safe Locks"]
        }
    ]
    triplets = create_triplets_for_products(data_list)

    fine_tune_sbert(
        triplets=triplets,
        base_model=cfg["base_model"],
        output_dir=cfg["output_dir"],
        experiment_name=cfg["experiment_name"],
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        model_registry_name=cfg["model_registry_name"],
        initial_stage="staging"  # Consistent with automation script
    )

    logger.info("Script completed. Check MLflow UI at http://127.0.0.1:5000.")
