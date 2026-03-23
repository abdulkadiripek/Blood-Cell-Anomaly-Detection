"""
Blood Cell Anomaly Detection - Preprocessing Pipeline
Handles feature encoding, scaling, and transformation to match training pipeline.
"""
import numpy as np
import pandas as pd
import pickle
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pathlib import Path


# Cell types from the training data (LabelEncoder alphabetical order)
CELL_TYPES = [
    'Acanthocyte', 'Band_Neutrophil', 'Basophil', 'Blast_Cell',
    'Burr_Cell', 'Eosinophil', 'Giant_Platelet',
    'Hypersegmented_Neutrophil', 'Lymphocyte', 'Macro_Ovalocyte',
    'Monocyte', 'Neutrophil', 'Normal_RBC', 'Platelet',
    'Schistocyte', 'Sickle_Cell', 'Spherocyte', 'Target_Cell',
    'Tear_Drop_Cell'
]

# Age group mapping
AGE_MAPPING = {
    'Pediatric': 0,
    'Adult': 1,
    'Elderly': 2
}

# Sex mapping (LabelEncoder alphabetical)
SEX_MAPPING = {
    'F': 0,
    'M': 1
}

# Feature order as used during training (after dropping unimportant cols & disease_category)
FEATURE_NAMES = [
    'cell_type', 'cell_diameter_um', 'nucleus_area_pct', 'chromatin_density',
    'cytoplasm_ratio', 'circularity', 'eccentricity', 'granularity_score',
    'lobularity_score', 'membrane_smoothness', 'cell_area_px', 'perimeter_px',
    'patient_age_group', 'patient_sex', 'wbc_count_per_ul',
    'rbc_count_millions_per_ul', 'hemoglobin_g_dl', 'hematocrit_pct',
    'platelet_count_per_ul', 'mcv_fl', 'mchc_g_dl'
]


def get_scaler_path():
    return os.path.join(os.path.dirname(__file__), 'scaler.pkl')


def fit_and_save_scaler(csv_path: str):
    """
    Fit StandardScaler on training data from the CSV and save it.
    This replicates the training pipeline exactly.
    """
    df = pd.read_csv(csv_path)

    # Drop unimportant columns (same as notebook)
    unimportant_columns = [
        'cell_id', 'anomaly_label', 'mean_r', 'mean_g', 'mean_b',
        'stain_intensity', 'dataset_source', 'staining_protocol',
        'microscope_model', 'magnification_x', 'image_resolution_px',
        'cytodiffusion_anomaly_score', 'cytodiffusion_classification_confidence',
        'labeller_confidence_score'
    ]
    df.drop(unimportant_columns, inplace=True, axis=1)

    # Save y (disease_category) before dropping
    y = df["disease_category"].values
    df = df.drop("disease_category", axis=1)

    # Map age groups
    df['patient_age_group'] = df['patient_age_group'].map(AGE_MAPPING)

    # Encode object columns with LabelEncoder
    le = LabelEncoder()
    object_columns = df.select_dtypes(include='object').columns
    for col in object_columns:
        df[col] = le.fit_transform(df[col])

    x = df.values

    # Split same as notebook (train_size=0.8, random_state=42, stratify=y)
    from sklearn.model_selection import train_test_split
    le_y = LabelEncoder()
    y_encoded = le_y.fit_transform(y)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y_encoded, train_size=0.8, random_state=42, stratify=y_encoded
    )

    # Fit scaler on training data only
    scaler = StandardScaler()
    scaler.fit(x_train)

    # Save scaler
    scaler_path = get_scaler_path()
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    return scaler


def load_scaler():
    """Load the saved StandardScaler."""
    scaler_path = get_scaler_path()
    if not os.path.exists(scaler_path):
        return None
    with open(scaler_path, 'rb') as f:
        return pickle.load(f)


def preprocess_input(data: dict, scaler: StandardScaler) -> np.ndarray:
    """
    Preprocess user input data to match the training pipeline.
    data should contain all feature values with proper keys.
    """
    # Encode cell_type
    cell_type_str = data.get('cell_type', 'Neutrophil')
    if cell_type_str in CELL_TYPES:
        cell_type_encoded = CELL_TYPES.index(cell_type_str)
    else:
        cell_type_encoded = 0

    # Encode patient_age_group
    age_group_str = data.get('patient_age_group', 'Adult')
    age_encoded = AGE_MAPPING.get(age_group_str, 1)

    # Encode patient_sex
    sex_str = data.get('patient_sex', 'M')
    sex_encoded = SEX_MAPPING.get(sex_str, 1)

    features = np.array([[
        cell_type_encoded,
        float(data.get('cell_diameter_um', 10.0)),
        float(data.get('nucleus_area_pct', 43.5)),
        float(data.get('chromatin_density', 0.39)),
        float(data.get('cytoplasm_ratio', 0.56)),
        float(data.get('circularity', 0.77)),
        float(data.get('eccentricity', 0.37)),
        float(data.get('granularity_score', 1.88)),
        float(data.get('lobularity_score', 1.77)),
        float(data.get('membrane_smoothness', 0.84)),
        int(float(data.get('cell_area_px', 1500))),
        int(float(data.get('perimeter_px', 150))),
        age_encoded,
        sex_encoded,
        int(float(data.get('wbc_count_per_ul', 7500))),
        float(data.get('rbc_count_millions_per_ul', 4.7)),
        float(data.get('hemoglobin_g_dl', 13.5)),
        float(data.get('hematocrit_pct', 41.0)),
        int(float(data.get('platelet_count_per_ul', 250000))),
        float(data.get('mcv_fl', 89.0)),
        float(data.get('mchc_g_dl', 33.5))
    ]])

    # Scale features
    scaled = scaler.transform(features)
    return scaled
