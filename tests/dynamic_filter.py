import re

def extract_price_range(query: str):
    """
    Extract a price range from the query.
    Examples:
    - "between 3000 and 6000" returns (3000, 6000)
    - "3000-6000" returns (3000, 6000)
    - "less than 2000", "under 2000", or "below 2000" returns (0, 2000)
    """
    # Pattern: "between 3000 and 6000" or "between 3000-6000" or "between 3000 to 6000"
    pattern = re.compile(r'between\s+(\d+)\s*(?:-|to|and)\s*(\d+)', re.IGNORECASE)
    match = pattern.search(query)
    if match:
        return float(match.group(1)), float(match.group(2))
    
    # Pattern: "3000-6000"
    pattern2 = re.compile(r'(\d+)\s*[-–]\s*(\d+)', re.IGNORECASE)
    match2 = pattern2.search(query)
    if match2:
        return float(match2.group(1)), float(match2.group(2))
    
    # Pattern: "less than 2000", "under 2000", or "below 2000"
    pattern3 = re.compile(r'(?:less than|under|below)\s+(\d+)', re.IGNORECASE)
    match3 = pattern3.search(query)
    if match3:
        max_price = float(match3.group(1))
        return 0, max_price  # assume no lower bound
    
    return None, None


def extract_rating_threshold(query: str):
    """
    Extract a rating threshold from the query.
    For instance, if the query contains "with high rating" or "rating above 4",
    we return a numeric threshold (e.g., 4.0).
    """
    # Look for "high rating" (default to 4.0)
    pattern = re.compile(r'(?:with\s+high\s+rating|high\s+rating)', re.IGNORECASE)
    if pattern.search(query):
        return 4.0
    
    # Look for explicit rating threshold, e.g., "rating above 4" or "rating over 4"
    pattern2 = re.compile(r'rating\s+(?:above|over|greater\s+than)\s*(\d+(\.\d+)?)', re.IGNORECASE)
    match = pattern2.search(query)
    if match:
        return float(match.group(1))
    
    return None

def numeric_rerank(products, query: str, default_min=0, default_max=float('inf')):
    """
    Re-ranks a list of product documents based on dynamic numeric constraints extracted from the query.
    
    - Filters products whose prices are within the extracted price range.
    - If a rating threshold is present, only considers products meeting that threshold.
    - Sorts in-range products by closeness to the midpoint of the price range and by descending rating.
    - Appends products that don't meet the criteria after the sorted ones.
    """
    min_price, max_price = extract_price_range(query)
    rating_threshold = extract_rating_threshold(query)
    if min_price is None or max_price is None:
        min_price, max_price = default_min, default_max

    # Use the midpoint as target for sorting (e.g., for "less than 2000", target is 1000)
    target = (min_price + max_price) / 2

    in_range = []
    out_range = []

    for product in products:
        try:
            price = float(product.metadata.get("price", 0))
            rating = float(product.metadata.get("rating", 0))
        except ValueError:
            price = 0
            rating = 0

        # First check if product falls in the price range
        if min_price <= price <= max_price:
            # If a rating threshold is set, only include products meeting the threshold in the in_range list.
            if rating_threshold is not None:
                if rating >= rating_threshold:
                    in_range.append((price, rating, product))
                else:
                    out_range.append(product)
            else:
                in_range.append((price, rating, product))
        else:
            out_range.append(product)
    
    # Sort in-range products first by closeness to the target price, then by descending rating.
    in_range.sort(key=lambda x: (abs(x[0] - target), -x[1]))
    sorted_in_range = [item[2] for item in in_range]

    return sorted_in_range + out_range