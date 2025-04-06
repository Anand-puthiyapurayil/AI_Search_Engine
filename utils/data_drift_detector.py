import os
import whylogs
from whylogs.core import DatasetProfileView
from whylogs.viz.drift.column_drift_algorithms import calculate_drift_scores

def save_reference_profile(profile_view: DatasetProfileView, path: str):
    """
    Saves a whylogs profile view (reference) to disk at 'path'.
    """
    profile_view.write(path)

def load_reference_profile(path: str) -> DatasetProfileView:
    """
    Loads an existing whylogs DatasetProfileView from disk.
    Raises FileNotFoundError if it doesn't exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"No reference profile found at: {path}")
    return DatasetProfileView.read(path)

def check_drift_against_reference(
    new_profile_view: DatasetProfileView,
    reference_profile_view: DatasetProfileView,
    with_thresholds: bool = True
):
    """
    Use whylogs' calculate_drift_scores to compare 'new' vs. 'reference'.
    Returns:
      - drift_detected (bool)
      - drift_scores (dict): all column-level drift info
    """
    drift_scores = calculate_drift_scores(
        target_view=new_profile_view,
        reference_view=reference_profile_view,
        with_thresholds=with_thresholds
    )

    # Instead of relying on "threshold_exceeded", check the "drift_category"
    # If any column has drift_category == "DRIFT", we consider that "drift_detected"

    drift_detected = False
    for col, info in drift_scores.items():
        if info.get("drift_category") == "DRIFT":
            drift_detected = True
            break

    return drift_detected, drift_scores
