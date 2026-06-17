# ============================================================
# src/models.py
# Définition et entraînement des modèles ML
# ============================================================

import numpy as np
import joblib
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_models() -> dict:
    """Retourne le dictionnaire de tous les modèles avec leurs paramètres par défaut."""
    return {
        'Régression Logistique': LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        ),
        'Arbre de Décision': DecisionTreeClassifier(
            max_depth=5,
            class_weight='balanced',
            random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            scale_pos_weight=3,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss',
            verbosity=0
        ),
        'LightGBM': LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
            verbose=-1
        ),
    }


def get_param_grids() -> dict:
    """Grilles d'hyperparamètres pour GridSearchCV."""
    return {
        'Random Forest': {
            'n_estimators': [100, 200],
            'max_depth': [5, 10, 15],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        },
        'XGBoost': {
            'n_estimators': [100, 200],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 1.0]
        },
        'LightGBM': {
            'n_estimators': [100, 200],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.01, 0.05, 0.1],
            'num_leaves': [31, 63]
        }
    }


def train_all_models(
    X_train_scaled: np.ndarray,
    X_train: any,
    y_train: any,
    models_dir: str = '../models/'
) -> dict:
    """
    Entraîne tous les modèles sur X_train_scaled (rééchantillonné par SMOTE).
    Tous les modèles utilisent les données scalées pour cohérence avec SMOTE.
    """
    models = get_models()
    trained = {}

    for name, model in models.items():
        logger.info(f"Entraînement : {name}...")
        model.fit(X_train_scaled, y_train)
        trained[name] = model

        safe_name = (name.lower()
                     .replace(' ', '_')
                     .replace('é', 'e')
                     .replace('è', 'e')
                     .replace('ê', 'e'))
        path = f"{models_dir}{safe_name}.pkl"
        joblib.dump(model, path)
        logger.info(f"  Sauvegardé : {path}")

    return trained


def optimize_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: any,
    models_dir: str = '../models/'
) -> any:
    """
    Optimise un modèle via GridSearchCV et le sauvegarde.
    X_train doit être X_train_scaled (données rééchantillonnées par SMOTE).
    """
    models = get_models()
    param_grids = get_param_grids()

    if model_name not in param_grids:
        logger.warning(f"Pas de grille définie pour {model_name}")
        return models[model_name]

    logger.info(f"GridSearchCV pour {model_name}...")
    grid = GridSearchCV(
        models[model_name],
        param_grids[model_name],
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=0
    )
    grid.fit(X_train, y_train)

    logger.info(f"  Meilleurs params : {grid.best_params_}")
    logger.info(f"  Meilleur AUC-ROC (CV) : {grid.best_score_:.4f}")

    safe_name = model_name.lower().replace(' ', '_') + '_best'
    path = f"{models_dir}{safe_name}.pkl"
    joblib.dump(grid.best_estimator_, path)
    logger.info(f"  Meilleur modèle sauvegardé : {path}")

    return grid.best_estimator_


def cross_validate_models(
    trained_models: dict,
    X_train_scaled: np.ndarray,
    X_train: any,
    y_train: any,
    cv: int = 5
) -> dict:
    """
    Effectue la validation croisée sur tous les modèles.
    Tous utilisent X_train_scaled pour cohérence avec SMOTE.
    """
    logger.info(f"Validation croisée {cv}-fold...")
    cv_results = {}

    for name, model in trained_models.items():
        scores = cross_val_score(
            model, X_train_scaled, y_train,
            cv=cv, scoring='roc_auc', n_jobs=-1
        )
        cv_results[name] = {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores.tolist()
        }
        logger.info(f"  {name} : {scores.mean():.4f} (+/- {scores.std():.4f})")

    return cv_results
