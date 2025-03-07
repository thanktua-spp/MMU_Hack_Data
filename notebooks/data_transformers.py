import pandas as pd
import numpy as np

# Replace None with Nan Transformer.
def replace_none_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    """Replaces None with np.nan in categorical columns."""
    df = df.copy()
    categorical_cols = df.select_dtypes(include=['object']).columns
    df[categorical_cols] = df[categorical_cols].apply(lambda col: col.map(lambda x: np.nan if x is None else x))
    return df


def drop_features(df: pd.DataFrame) -> pd.DataFrame:
    """Drop features not required"""
    df = df.copy()
    columns_to_drop = ["stock_item_id", "last_date_seen", "first_date_seen", "derivative_id", "first_registration_date"]
    return df.drop(columns=columns_to_drop, errors='ignore')


def convert_columns_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Converts specified object columns to float/numeric type without dropping them."""
    df = df.copy()
    
    convert_columns = ["zero_to_sixty_mph_seconds", "engine_power_bhp", "fuel_economy_wltp_combined_mpg",
                       "battery_usable_capacity_kwh", "length_mm", "insurance_group", "plate"]
    
    # Ensure we only select columns that exist in the DataFrame
    existing_columns = list(set(df.columns) & set(convert_columns))
    
    if existing_columns:  # Only apply conversion if columns exist
        df[existing_columns] = df[existing_columns].apply(pd.to_numeric, errors='coerce')

    return df


def compute_score(y_true, y_pred):
    return {
        "R2": f"{r2_score(y_true, y_pred):.3f}",
        "MedAE": f"{median_absolute_error(y_true, y_pred):.3f}",
    }