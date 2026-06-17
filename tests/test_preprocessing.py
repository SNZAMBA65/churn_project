# ============================================================
# tests/test_preprocessing.py
# Tests unitaires du module de prétraitement
# ============================================================

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import pytest
from preprocessing import clean_data, encode_features


@pytest.fixture
def sample_raw_df():
    """Crée un mini dataset de test similaire au dataset Telco."""
    return pd.DataFrame({
        'customerID': ['0001-AAA', '0002-BBB', '0003-CCC'],
        'gender': ['Male', 'Female', 'Male'],
        'SeniorCitizen': [0, 1, 0],
        'Partner': ['Yes', 'No', 'Yes'],
        'Dependents': ['No', 'No', 'Yes'],
        'tenure': [0, 24, 5],
        'PhoneService': ['Yes', 'Yes', 'No'],
        'MultipleLines': ['No', 'Yes', 'No phone service'],
        'InternetService': ['DSL', 'Fiber optic', 'No'],
        'OnlineSecurity': ['No', 'Yes', 'No internet service'],
        'OnlineBackup': ['No', 'Yes', 'No internet service'],
        'DeviceProtection': ['No', 'Yes', 'No internet service'],
        'TechSupport': ['No', 'Yes', 'No internet service'],
        'StreamingTV': ['No', 'Yes', 'No internet service'],
        'StreamingMovies': ['No', 'Yes', 'No internet service'],
        'Contract': ['Month-to-month', 'Two year', 'Month-to-month'],
        'PaperlessBilling': ['Yes', 'No', 'Yes'],
        'PaymentMethod': ['Electronic check', 'Mailed check', 'Electronic check'],
        'MonthlyCharges': [29.85, 56.95, 20.0],
        'TotalCharges': [' ', '1889.5', '100.0'],  # espace vide volontaire
        'Churn': ['No', 'No', 'Yes']
    })


def test_clean_data_removes_customer_id(sample_raw_df):
    """Vérifie que customerID est bien supprimé."""
    df_clean = clean_data(sample_raw_df)
    assert 'customerID' not in df_clean.columns


def test_clean_data_fixes_total_charges(sample_raw_df):
    """Vérifie que TotalCharges devient numérique sans NaN."""
    df_clean = clean_data(sample_raw_df)
    assert df_clean['TotalCharges'].dtype in ['float64', 'int64']
    assert df_clean['TotalCharges'].isnull().sum() == 0


def test_clean_data_handles_empty_total_charges(sample_raw_df):
    """Vérifie que la valeur vide (tenure=0) devient 0."""
    df_clean = clean_data(sample_raw_df)
    assert df_clean.loc[0, 'TotalCharges'] == 0.0


def test_clean_data_converts_senior_citizen(sample_raw_df):
    """Vérifie que SeniorCitizen passe de 0/1 à No/Yes."""
    df_clean = clean_data(sample_raw_df)
    assert set(df_clean['SeniorCitizen'].unique()).issubset({'Yes', 'No'})


def test_encode_features_produces_numeric_columns(sample_raw_df):
    """Vérifie que toutes les colonnes sont numériques après encodage."""
    df_clean = clean_data(sample_raw_df)
    df_encoded = encode_features(df_clean)
    non_numeric = df_encoded.select_dtypes(exclude=['number', 'bool']).columns.tolist()
    assert len(non_numeric) == 0, f"Colonnes non numériques restantes : {non_numeric}"


def test_encode_features_keeps_same_row_count(sample_raw_df):
    """Vérifie qu'aucune ligne n'est perdue pendant l'encodage."""
    df_clean = clean_data(sample_raw_df)
    df_encoded = encode_features(df_clean)
    assert len(df_encoded) == len(df_clean)