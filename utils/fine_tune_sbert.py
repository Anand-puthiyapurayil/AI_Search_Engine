import os
import random
import itertools
from collections import defaultdict
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
from utils.utils import load_config, load_dataframe
from logger.logger import get_logger

logger = get_logger(__file__)

##############################
# Pair Generation Functions  #
##############################

# Helper: safely convert a value to lowercase (handles NaN)
def safe_lower(value):
    if pd.isna(value):
        return ''
    return str(value).lower().strip()

# Helper: build a text representation for an item
def build_text(item):
    return (
        f"product Name: {item.get('product_name', 'N/A')}\n"
        f"supplier: {item.get('supplier_name', 'N/A') if not pd.isna(item.get('supplier_name')) else 'N/A'}\n"
        f"city: {item.get('city', 'N/A')}\n"
        f"state: {item.get('state', 'N/A')}\n"
        f"country: {item.get('country', 'N/A')}\n"
        f"category: {item.get('category', 'N/A')}\n"
        f"subcategory: {item.get('subcategory', 'N/A')}\n"
        f"sub-subcategory: {item.get('subsubcategory', 'N/A') if not pd.isna(item.get('subsubcategory')) else 'N/A'}\n"
        f"variant: {item.get('variation', 'N/A') if not pd.isna(item.get('variation')) else 'N/A'}"
    )

# Positive pairs: group items by (product name, category, subcategory, city, state, country)
def create_positive_pairs(data_list, max_examples=10000):
    pair_examples = []
    product_groups = defaultdict(list)
    
    for item in data_list:
        key = (
            safe_lower(item.get('product_name', '')),
            safe_lower(item.get('category', '')),
            safe_lower(item.get('subcategory', '')),
            safe_lower(item.get('city', '')),
            safe_lower(item.get('state', '')),
            safe_lower(item.get('country', ''))
        )
        product_groups[key].append(item)
    
    for group in product_groups.values():
        if len(group) < 2:
            continue
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                text1 = build_text(group[i])
                text2 = build_text(group[j])
                pair_examples.append(InputExample(texts=[text1, text2], label=1.0))
                if len(pair_examples) >= max_examples:
                    return pair_examples
    return pair_examples

# Generate negative pairs by sampling items from different product groups.
def create_negative_pairs(data_list, max_examples=10000):
    negative_examples = []
    product_groups = defaultdict(list)
    
    for item in data_list:
        key = (
            safe_lower(item.get('product_name', '')),
            safe_lower(item.get('category', '')),
            safe_lower(item.get('subcategory', '')),
            safe_lower(item.get('city', '')),
            safe_lower(item.get('state', '')),
            safe_lower(item.get('country', ''))
        )
        product_groups[key].append(item)
    
    keys = list(product_groups.keys())
    while len(negative_examples) < max_examples:
        key1, key2 = random.sample(keys, 2)
        group1 = product_groups[key1]
        group2 = product_groups[key2]
        item1 = random.choice(group1)
        item2 = random.choice(group2)
        text1 = build_text(item1)
        text2 = build_text(item2)
        negative_examples.append(InputExample(texts=[text1, text2], label=0.0))
    return negative_examples

# Explicit negative examples provided manually.
def add_explicit_negative_examples(negative_examples):
    explicit_negatives = [
        InputExample(
            texts=[
                "product Name: engine oil\nsupplier: dhofar\ncity: muscat\nstate: muscat governorate\nCountry: oman\ncategory: chemicals and minerals\nsubcategory: fuels and lubricants\nSub-subcategory: lubricants and waxes\nVariant: high performance",
                "product Name: olive oil\nsupplier: lulu arabia llc\ncity: dubai\nstate: dhofar governorate\nCountry: uae\ncategory: chemicals and minerals\nsubcategory: fuels and lubricants\nSub-subcategory: lubricants and waxes\nVariant: standard"
            ],
            label=0.0
        ),
        InputExample(
            texts=[
                "product Name: engine oil\nsupplier: well head component inc avsco houston\ncity: missouri\nstate: texas\nCountry: oman\ncategory: chemicals and minerals\nsubcategory: fuels and lubricants\nSub-subcategory: lubricants and waxes\nVariant: small",
                "product Name: soya oil\nsupplier: dhofar environmental safety services co llc\ncity: salalah\nstate: dhofar governorate\nCountry: oman\ncategory: food products and beverages\nsubcategory: food products\nSub-subcategory: edible oils and fats\nVariant: large"
            ],
            label=0.0
        ),
        InputExample(
            texts=[
                "product Name: carbon\nsupplier: the national detergent co saog\ncity: bowsher\nstate: muscat governorate\nCountry: oman\ncategory: chemicals and minerals\nsubcategory: application gases\nSub-subcategory: carbon\nVariant: large",
                "product Name: carbon watercan\nsupplier: steel tubes india\ncity: mumbai\nstate: maharashtra\nCountry: oman\ncategory: general mechanical equipment and materials\nsubcategory: transmission equipment and accessories\nSub-subcategory: couplings\nVariant: large"
            ],
            label=0.0
        )
    ]
    negative_examples.extend(explicit_negatives)
    return negative_examples

##############################################
# Fine-Tuning & MLflow Model Training Section #
##############################################

def fine_tune_contrastive(
    positive_pairs,
    negative_pairs,
    base_model,
    output_dir,
    epochs,
    batch_size,
    learning_rate,
    experiment_name,
    model_registry_name,
    initial_stage="staging"
):
    # Configure MLflow (tracking URI, experiment, etc.)
    mlflow.set_tracking_uri("http://192.168.1.227:5000")
    client = MlflowClient()
    mlflow.set_experiment(experiment_name)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer(base_model, device=device)
    logger.info(f"Using device: {device}")

    # Combine positive and negative pairs, then shuffle.
    contrastive_examples = positive_pairs + negative_pairs
    random.shuffle(contrastive_examples)
    dataloader = DataLoader(contrastive_examples, shuffle=True, batch_size=batch_size)
    loss_fn = losses.ContrastiveLoss(model=model)
    warmup_steps = int(len(dataloader) * epochs * 0.1)
    
    logger.info("Starting fine-tuning with CosineLoss...")

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        mlflow.log_param("base_model", base_model)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", learning_rate)
        
        model.fit(
            train_objectives=[(dataloader, loss_fn)],
            epochs=epochs,
            warmup_steps=warmup_steps,
            output_path=output_dir,
            optimizer_params={'lr': learning_rate}
        )
        
        mlflow.log_metric("training_completed", 1)
        logged_model = mlflow.sentence_transformers.log_model(model=model, artifact_path="sbert_model")
        
        try:
            client.create_registered_model(model_registry_name)
        except MlflowException as e:
            if "RESOURCE_ALREADY_EXISTS" not in str(e):
                raise e
                
        model_version = client.create_model_version(
            name=model_registry_name,
            source=logged_model.model_uri,
            run_id=run_id
        )
        if model_version:
            client.set_model_version_tag(
                name=model_registry_name,
                version=model_version.version,
                key="stage",
                value=initial_stage
            )

    logger.info("Contrastive fine-tuning completed. Model saved to " + output_dir)

#########################################
# Main Pipeline Execution               #
#########################################

def main():
    try:
        # Load configuration and data
        this_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(this_dir, "..", "config.yaml")
        config = load_config(config_path)

        df = load_dataframe(config["product_data"])

        # If you want to use the full data, you can also do:
        data_list = df.to_dict(orient="records")
        
        # Generate positive and negative pairs per revised criteria.
        positive_pairs = create_positive_pairs(data_list, max_examples=10000)
        negative_pairs = create_negative_pairs(data_list, max_examples=10000)
        # Add explicit negative examples.
        negative_pairs = add_explicit_negative_examples(negative_pairs)
        
        logger.info(f"Generated {len(positive_pairs)} positive pairs.")
        logger.info(f"Generated {len(negative_pairs)} negative pairs.")
        
        # Fine-tune the SentenceTransformer model using MLflow
        fine_tune_contrastive(
            positive_pairs=positive_pairs,
            negative_pairs=negative_pairs,
            base_model=config["base_model"],
            output_dir=config["output_dir2"],
            epochs=config.get("epochs", 15),
            batch_size=config.get("batch_size", 32),
            learning_rate=float(config.get("learning_rate", 2e-5)),
            experiment_name=config["experiment_name"],
            model_registry_name=config["model_registry_name"],
            initial_stage=config.get("initial_stage", "staging")
        )
        
        logger.info("MLflow Contrastive fine-tuning completed successfully.")
    except Exception as e:
        logger.error(f"Error in main: {e}", exc_info=True)
        raise e

if __name__ == "__main__":
    main()
