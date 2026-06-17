# ============================================================
# src/preprocessing.py
# Pipeline de prétraitement des données
# ============================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_raw_data(path: str) -> pd.DataFrame:
    """Charge le dataset brut depuis le chemin spécifié."""
    logger.info(f"Chargement des données depuis {path}")
    df = pd.read_csv(path)
    logger.info(f"Dataset chargé : {df.shape[0]} lignes x {df.shape[1]} colonnes")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie le dataset brut :
    - Corrige TotalCharges (str -> float)
    - Harmonise SeniorCitizen (0/1 -> Yes/No)
    - Supprime customerID
    """
    logger.info("Nettoyage des données...")
    df = df.copy()

    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    n_nan = df['TotalCharges'].isnull().sum()
    if n_nan > 0:
        logger.info(f"  {n_nan} valeurs manquantes dans TotalCharges (tenure=0) -> remplacées par 0")
        df['TotalCharges'] = df['TotalCharges'].fillna(0)

    if df['SeniorCitizen'].dtype in ['int64', 'int32']:
        df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})

    n_dup = df.duplicated().sum()
    logger.info(f"  Doublons détectés : {n_dup}")

    logger.info(f"Nettoyage terminé : {df.shape[0]} lignes x {df.shape[1]} colonnes")
    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode les variables catégorielles :
    - Label Encoding pour les binaires Yes/No
    - One-Hot Encoding pour les variables à 3+ modalités
    """
    logger.info("Encodage des variables catégorielles...")
    df = df.copy()

    binary_cols = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents',
        'PhoneService', 'PaperlessBilling', 'Churn'
    ]
    binary_map = {'Yes': 1, 'No': 0, 'Female': 0, 'Male': 1}
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map(binary_map)

    multi_cols = [
        'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod'
    ]
    df = pd.get_dummies(df, columns=multi_cols, drop_first=True)

    logger.info(f"Encodage terminé : {df.shape[1]} colonnes")
    return df


def split_and_scale(
    df: pd.DataFrame,
    target: str = 'Churn',
    test_size: float = 0.2,
    apply_smote: bool = True,
    scaler_path: str = '../models/scaler.pkl',
    feature_names_path: str = '../models/feature_names.pkl'
) -> tuple:
    """
    Découpe train/test, normalise et applique SMOTE si demandé.
    Retourne : X_train_raw, X_test, X_train_resampled, X_test_scaled, y_train_resampled, y_test

    Note : SMOTE est appliqué sur les données scalées uniquement.
    Après SMOTE, X_train_resampled et y_train_resampled sont cohérents.
    Tous les modèles utilisent ensuite X_train_resampled.
    """
    logger.info("Séparation train/test et normalisation...")

    X = df.drop(columns=[target])
    y = df[target]

    joblib.dump(X.columns.tolist(), feature_names_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    logger.info(f"  Train : {X_train.shape[0]} | Test : {X_test.shape[0]}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, scaler_path)
    logger.info(f"  Scaler sauvegardé : {scaler_path}")

    if apply_smote:
        logger.info("  Application de SMOTE pour rééquilibrer les classes...")
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
        logger.info(f"  Après SMOTE : {pd.Series(y_train_resampled).value_counts().to_dict()}")
    else:
        X_train_resampled = X_train_scaled
        y_train_resampled = y_train

    return X_train, X_test, X_train_resampled, X_test_scaled, y_train_resampled, y_test


def run_preprocessing_pipeline(
    raw_path: str = '../data/WA_Fn-UseC_-Telco-Customer-Churn.csv',
    export_path: str = '../exports/dataset_clean.csv'
) -> tuple:
    """Pipeline complet : charge, nettoie, encode, split, scale."""
    df_raw = load_raw_data(raw_path)
    df_clean = clean_data(df_raw)
    df_clean.to_csv(export_path, index=False)
    logger.info(f"Dataset propre exporté : {export_path}")
    df_encoded = encode_features(df_clean)
    return split_and_scale(df_encoded)