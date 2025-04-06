import os
import re
import pandas as pd
import numpy as np
from typing import Tuple, Dict
import sys
# 1) Standard imports from your existing pipeline
from utils.utils import load_config
from sqlalchemy import create_engine
from dotenv import load_dotenv
from logger.logger import get_logger

# 2) Import your drift functions from data_drift_detector
import whylogs
from utils.data_drift_detector import (
    save_reference_profile,
    load_reference_profile,
    check_drift_against_reference
)

logger = get_logger(__file__)
load_dotenv()

# Paths to store references for each subset
PRODUCTS_REF_PATH = "products_profile.bin"
SERVICES_REF_PATH = "services_profile.bin"
SHOPS_REF_PATH = "shops_profile.bin"

def create_db_engine(config: dict):
    db_config = {
    }
    connection_string = (
        f"mysql+pymysql://{db_config['user']}:{db_config['password']}"
        f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
    )
    engine = create_engine(connection_string)
    logger.info(
        f"Database engine created for {db_config['host']}:{db_config['port']}/"
        f"{db_config['database']}"
    )
    return engine

SQL_QUERY = """
SELECT 
    ps.id as unique_id,
    p.name AS display_name, 
    p.description, 
    ps.price, 
    p.discount, 
    p.rating, 
    p.digital, 
    p.bsource AS business_source, 
    ps.variant, 
    u.id AS user_id, 
    u.name AS supplier, 
    u.country, 
    u.stateName AS state, 
    u.city, 
    u.classification AS class, 
    u.ClassDisplay AS classification, 
    u.shortDesc AS store_description,
    s.shop_id AS store_id, 
    s.shop_name AS store, 
    COALESCE(bgp.bicpm_productname, bgs.bicsm_servicename) AS subsubcategory_name, 
    GROUP_CONCAT(DISTINCT sub_cat.bicsc_subcategoryname) AS subcategory_names, 
    GROUP_CONCAT(DISTINCT cat.bicc_categoryname) AS category_names
FROM 
    product_stocks ps
LEFT JOIN products p ON ps.product_id = p.id AND ps.product_id IS NOT NULL
LEFT JOIN users u ON p.user_id = u.id
LEFT JOIN (
    SELECT user_id, MAX(id) AS shop_id, MAX(name) AS shop_name
    FROM shops
    GROUP BY user_id
) s ON s.user_id = u.id
LEFT JOIN bgiinduscodeprodmst_tbl bgp 
    ON p.subsubcategory_id = bgp.bgiinduscodeprodmst_pk AND p.digital = 0
LEFT JOIN bgiinduscodeservmst_tbl bgs 
    ON p.subsubcategory_id = bgs.bgiinduscodeservmst_pk AND p.digital = 1
LEFT JOIN bgiindcodesubcateg_tbl sub_cat 
    ON sub_cat.bicsc_bgiindcodecateg_fk = p.subcategory_id
LEFT JOIN bgiindcodecateg_tbl cat 
    ON cat.bgiindcodecateg_pk = p.category_id
WHERE 
    p.name IS NOT NULL
GROUP BY 
    ps.product_id, p.name, p.description, p.unit_price, p.discount, 
    p.rating, p.digital, p.bsource, p.prod_ind_name, ps.variant, 
    u.id, u.name, u.country, u.stateName, u.city, 
    u.classification, u.ClassDisplay, s.shop_id, s.shop_name, 
    COALESCE(bgp.bicpm_productname, bgs.bicsm_servicename);
"""

def fetch_data(query: str, engine, chunksize=10000) -> pd.DataFrame:
    logger.info("Connecting to database and fetching data...")
    chunks = []
    try:
        with engine.connect() as connection:
            for i, chunk in enumerate(pd.read_sql(query, connection, chunksize=chunksize)):
                chunks.append(chunk)
                logger.debug(f"Fetched chunk {i+1} with {len(chunk)} rows")
        data = pd.concat(chunks, ignore_index=True)
        logger.info(f"Data fetched successfully. Total rows: {len(data)}")
        return data
    except Exception as e:
        logger.error(f"Error fetching data: {str(e)}")
        raise

def analyze_data(df: pd.DataFrame) -> Dict:
    stats = {
        'missing_values': df.isnull().sum().to_dict(),
        'dtypes': df.dtypes.to_dict(),
        'outliers': {},
        'unique_counts': df.nunique().to_dict(),
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2
    }
    for col in df.select_dtypes(include=['float64', 'int64']):
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
        stats['outliers'][col] = len(outliers)
    return stats

def clean_text(text: str) -> str:
    if pd.isnull(text):
        return ""
    import re
    patterns = {
        'html': re.compile(r'<[^>]+>'),
        'special_chars': re.compile(r'[^a-zA-Z0-9\s]'),
        'whitespace': re.compile(r'\s+')
    }
    text = patterns['html'].sub('', text)
    text = patterns['special_chars'].sub('', text)
    text = patterns['whitespace'].sub(' ', text).strip()
    return text.lower()

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    initial_stats = analyze_data(df)
    
    text_columns = ['description', 'variant', 'store', 'store_description']
    df[text_columns] = df[text_columns].apply(lambda col: col.apply(clean_text))
    
    # Fill numeric/categorical nulls
    df['price'] = df['price'].fillna(0)
    df['country'] = df['country'].fillna("0")
    df['state'] = df['state'].fillna("0")
    df['city'] = df['city'].fillna("0")

    # Ensure price is > 0
    df = df[df['price'] > 0]
    
    # Lowercase for object columns
    obj_cols = df.select_dtypes(include=['object']).columns
    df[obj_cols] = df[obj_cols].apply(lambda col: col.str.lower())
    
    final_stats = analyze_data(df)
    df.attrs['preprocessing_stats'] = {
        'initial': initial_stats,
        'final': final_stats,
        'rows_processed': len(df),
        'columns_processed': len(df.columns)
    }
    return df

def data_quality_checks(df: pd.DataFrame):
    issues = []
    critical_columns = ['display_name', 'price']
    for col in critical_columns:
        if df[col].isnull().any():
            issues.append(f"Column '{col}' contains missing values.")

    if df['display_name'].isnull().any():
        issues.append("Missing values found in 'display_name'.")
    if df['price'].isnull().any() or (df['price'] <= 0).any():
        issues.append("Invalid or missing values found in 'price'. Must be > 0.")
    if not pd.api.types.is_numeric_dtype(df['price']):
        issues.append("'price' column must be numeric.")
    if df['unique_id'].duplicated().any():
        issues.append("Duplicate values found in 'unique_id'.")
    return issues

def validation_checks(products_df: pd.DataFrame, services_df: pd.DataFrame, shops_df: pd.DataFrame):
    issues = []
    # Products
    if products_df.empty:
        issues.append("Products dataframe is empty.")
    if (products_df['product_name'] == "").any():
        issues.append("Products dataframe has empty 'product_name'.")
    if (products_df['price'] <= 0).any():
        issues.append("Products dataframe has invalid prices (<= 0).")

    # Services
    if services_df.empty:
        issues.append("Services dataframe is empty.")
    if (services_df['service_name'] == "").any():
        issues.append("Services dataframe has empty 'service_name'.")

    # Shops
    if shops_df.empty:
        issues.append("Shops dataframe is empty.")
    if shops_df['store'].isnull().any():
        issues.append("Shop dataframe has missing 'store' values.")
    if shops_df['store_id'].duplicated().any():
        issues.append("Duplicate store IDs found in shop dataframe.")
    
    return issues

def preprocess_and_separate(df: pd.DataFrame, data_dir: str):
    os.makedirs(data_dir, exist_ok=True)

    config_map = {
        'products': {
            'mask': df['digital'] == 0,
            'id_col': 'product_id',
            'name_col': 'product_name'
        },
        'services': {
            'mask': df['digital'] == 1,
            'id_col': 'service_id',
            'name_col': 'service_name'
        }
    }
    drop_cols = ['digital', 'user_id', 'discount', 'class', 'store_id', 'store_description']
    base_rename = {
        'unique_id': None,
        'display_name': None,
        'category_names': 'category',
        'subcategory_names': 'subcategory',
        'subsubcategory_name': 'subsubcategory'
    }

    processed_dfs = {}
    for df_type, params in config_map.items():
        rename_map = base_rename.copy()
        rename_map['unique_id'] = params['id_col']
        rename_map['display_name'] = params['name_col']
        
        subset = (
            df[params['mask']]
            .drop(columns=drop_cols)
            .rename(columns=rename_map)
            .reset_index(drop=True)
        )
        processed_dfs[df_type] = subset
    
    # Shops
    shop_columns = {
        'store': 'first',
        'store_id': 'first',
        'variant': lambda x: ', '.join(filter(bool, x)),
        'store_description': 'first',
        'supplier': 'first',
        'country': 'first',
        'state': 'first',
        'city': 'first',
        'business_source': 'first'
    }
    
    df_for_shops = df[['store', 'display_name', 'digital']].copy()
    df_for_shops['product_name'] = df_for_shops.apply(lambda x: x['display_name'] if x['digital'] == 0 else "", axis=1)
    df_for_shops['service_name'] = df_for_shops.apply(lambda x: x['display_name'] if x['digital'] == 1 else "", axis=1)
    df_for_shops.drop(['display_name', 'digital'], axis=1, inplace=True)
    
    shops_base_df = df[list(shop_columns.keys())].copy()
    
    shops_agg = df_for_shops.groupby('store', as_index=False).agg({
        'product_name': lambda x: ', '.join(filter(bool, x)),
        'service_name': lambda x: ', '.join(filter(bool, x))
    })
    shops_df = pd.merge(
        shops_base_df.groupby('store', as_index=False).agg(shop_columns),
        shops_agg,
        on='store',
        how='left'
    )
    
    dfs = {
        'products': processed_dfs['products'],
        'services': processed_dfs['services'],
        'shops': shops_df
    }

    for name, df_obj in dfs.items():
        df_obj.attrs['preprocessing_info'] = {
            'rows': len(df_obj),
            'columns': list(df_obj.columns),
            'memory_usage_mb': df_obj.memory_usage(deep=True).sum() / 1024**2
        }
        fname = "shop_data.csv" if name == 'shops' else f"{name}_data.csv"
        fpath = os.path.join(data_dir, fname)
        df_obj.to_csv(fpath, index=False)
        print(f"Saved {len(df_obj)} {name} to '{fpath}'")

    return processed_dfs['products'], processed_dfs['services'], shops_df

def main():
    drift_found = False  # Initialize a flag to track drift
    try:
        this_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(this_dir, "..", "config.yaml")
        config = load_config(config_path)
        logger.info(f"Configuration loaded from {config_path}")

        engine = create_db_engine(config)
        raw_data = fetch_data(SQL_QUERY, engine)

        # Preprocess entire dataset
        clean_data = preprocess_data(raw_data)

        # Data Quality on entire dataset
        quality_issues = data_quality_checks(clean_data)
        if quality_issues:
            logger.warning("Data Quality Issues:")
            for issue in quality_issues:
                logger.warning(f" - {issue}")
            raise ValueError("Data quality issues must be resolved.")

        # Separate into products, services, shops
        data_dir = config.get('data_dir', 'data/')
        products_df, services_df, shops_df = preprocess_and_separate(clean_data, data_dir)

        # Process each subset for drift detection
        subsets = [
            ("products", products_df, PRODUCTS_REF_PATH),
            ("services", services_df, SERVICES_REF_PATH),
            ("shops", shops_df, SHOPS_REF_PATH),
        ]

        for name, subset_df, profile_path in subsets:
            new_profile_view = whylogs.log(pandas=subset_df).profile().view()

            if os.path.exists(profile_path):
                logger.info(f"Reference profile found for {name}. Checking drift...")
                ref_profile_view = load_reference_profile(profile_path)

                drift_detected, drift_scores = check_drift_against_reference(
                    new_profile_view=new_profile_view,
                    reference_profile_view=ref_profile_view,
                    with_thresholds=True
                )

                # Check each column's drift status
                drift_cols = []
                for col, info in drift_scores.items():
                    if info.get("drift_category", "NO_DRIFT") == "DRIFT":
                        drift_cols.append(col)

                if drift_cols:
                    logger.warning(f"!!! DATA DRIFT DETECTED FOR {name.upper()} !!!")
                    logger.warning(f"Columns flagged as DRIFT: {drift_cols}")
                    drift_found = True
                    # Replace the reference with the new profile
                    logger.warning(f"Replacing {name} reference with the new profile, since drift was found.")
                    save_reference_profile(new_profile_view, profile_path)
                else:
                    logger.info(f"No significant drift detected for {name}.")
            else:
                logger.info(f"No reference for {name}. Saving this run as reference.")
                save_reference_profile(new_profile_view, profile_path)

        # Validation checks on parted data
        validation_issues = validation_checks(products_df, services_df, shops_df)
        if validation_issues:
            logger.warning("Validation Issues:")
            for issue in validation_issues:
                logger.warning(f" - {issue}")
            raise ValueError("Validation issues must be resolved.")

        logger.info("Pipeline completed successfully!")

        # Exit with non-zero code if drift was detected, else exit with 0.
        if drift_found:
            logger.info("Drift detected; exiting with non-zero exit code to trigger downstream pipelines.")
            sys.exit(1)
        else:
            logger.info("No drift detected; exiting with code 0.")
            sys.exit(0)

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(2)  # Use a different non-zero code for errors

if __name__ == "__main__":
    main()
