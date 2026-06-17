# ============================================================
# src/evaluation.py
# Métriques et visualisations d'évaluation des modèles
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve,
    confusion_matrix
)
from sklearn.model_selection import learning_curve
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams['figure.dpi'] = 100


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Modèle"
) -> dict:
    """Calcule toutes les métriques pour un modèle."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        'Modèle': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred),
        'AUC-ROC': roc_auc_score(y_test, y_proba),
        'y_pred': y_pred,
        'y_proba': y_proba
    }

    logger.info(
        f"{model_name} -> AUC: {metrics['AUC-ROC']:.4f} | "
        f"F1: {metrics['F1']:.4f} | Recall: {metrics['Recall']:.4f}"
    )
    return metrics


def evaluate_all_models(
    trained_models: dict,
    X_test_scaled: np.ndarray,
    X_test: any,
    y_test: any
) -> tuple:
    """
    Évalue tous les modèles sur X_test_scaled.
    Tous les modèles utilisent X_test_scaled pour cohérence.
    """
    results = []
    predictions = {}

    for name, model in trained_models.items():
        m = evaluate_model(model, X_test_scaled, y_test, name)
        predictions[name] = {
            'y_pred': m.pop('y_pred'),
            'y_proba': m.pop('y_proba')
        }
        results.append(m)

    df_results = pd.DataFrame(results).round(4)
    return df_results, predictions


def plot_roc_curves(
    trained_models: dict,
    X_test_scaled: np.ndarray,
    X_test: any,
    y_test: any,
    save_path: str = None
):
    """Trace les courbes ROC comparatives."""
    colors = ['#3498db', '#e67e22', '#9b59b6', '#2ecc71', '#e74c3c', '#1abc9c', '#f39c12']
    fig, ax = plt.subplots(figsize=(9, 7))

    for i, (name, model) in enumerate(trained_models.items()):
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, label=f'{name} (AUC={auc:.4f})',
                color=colors[i % len(colors)], linewidth=2.5)

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Aléatoire (AUC=0.5)')
    ax.set_xlabel('Taux de Faux Positifs', fontsize=12)
    ax.set_ylabel('Taux de Vrais Positifs', fontsize=12)
    ax.set_title('Courbes ROC - Comparaison des Modèles', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Courbes ROC sauvegardées : {save_path}")
    plt.show()


def plot_confusion_matrices(
    trained_models: dict,
    predictions: dict,
    y_test: any,
    save_path: str = None
):
    """Trace les matrices de confusion côte à côte."""
    n = len(trained_models)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))

    if n == 1:
        axes = [axes]

    for i, name in enumerate(trained_models.keys()):
        cm = confusion_matrix(y_test, predictions[name]['y_pred'])
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            ax=axes[i], linewidths=0.5,
            xticklabels=['Resté', 'Churné'],
            yticklabels=['Resté', 'Churné']
        )
        short_name = name.replace('Régression ', 'Rég.\n').replace(' ', '\n')
        axes[i].set_title(short_name, fontsize=11, fontweight='bold')
        axes[i].set_xlabel('Prédit')
        axes[i].set_ylabel('Réel')

    plt.suptitle('Matrices de Confusion', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Matrices sauvegardées : {save_path}")
    plt.show()


def plot_learning_curve(
    model,
    X_train: np.ndarray,
    y_train: any,
    model_name: str,
    save_path: str = None
):
    """Trace la courbe d'apprentissage pour détecter over/underfitting."""
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train,
        cv=5, scoring='roc_auc',
        train_sizes=np.linspace(0.1, 1.0, 10),
        n_jobs=-1
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(train_sizes, train_mean, 'o-', color='#3498db',
            label='Score entraînement', linewidth=2)
    ax.fill_between(train_sizes,
                    train_mean - train_std,
                    train_mean + train_std,
                    alpha=0.15, color='#3498db')
    ax.plot(train_sizes, val_mean, 'o-', color='#2ecc71',
            label='Score validation (CV)', linewidth=2)
    ax.fill_between(train_sizes,
                    val_mean - val_std,
                    val_mean + val_std,
                    alpha=0.15, color='#2ecc71')

    ax.set_xlabel("Taille du jeu d'entraînement", fontsize=12)
    ax.set_ylabel('AUC-ROC', fontsize=12)
    ax.set_title(f"Courbe d'apprentissage - {model_name}",
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 1.05)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Learning curve sauvegardée : {save_path}")
    plt.show()


def find_optimal_threshold(
    model,
    X_test: np.ndarray,
    y_test: any,
    model_name: str,
    save_path: str = None
) -> float:
    """
    Trouve le seuil de décision optimal qui maximise le F1-score.
    Par défaut sklearn utilise 0.5, mais ce n'est pas toujours optimal.
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    thresholds = np.arange(0.1, 0.9, 0.01)
    f1_scores = []
    recall_scores = []
    precision_scores = []

    for t in thresholds:
        y_pred_t = (y_proba >= t).astype(int)
        f1_scores.append(f1_score(y_test, y_pred_t))
        recall_scores.append(recall_score(y_test, y_pred_t))
        precision_scores.append(precision_score(y_test, y_pred_t))

    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(thresholds, f1_scores, label='F1-score',
            color='#3498db', linewidth=2.5)
    ax.plot(thresholds, recall_scores, label='Recall',
            color='#e74c3c', linewidth=2, linestyle='--')
    ax.plot(thresholds, precision_scores, label='Precision',
            color='#2ecc71', linewidth=2, linestyle='--')
    ax.axvline(x=optimal_threshold, color='#f39c12', linewidth=2,
               linestyle=':', label=f'Seuil optimal = {optimal_threshold:.2f}')
    ax.axvline(x=0.5, color='#95a5a6', linewidth=1.5,
               linestyle=':', label='Seuil par défaut = 0.50')

    ax.set_xlabel('Seuil de décision', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'Optimisation du seuil de décision - {model_name}',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Seuil optimal sauvegardé : {save_path}")
    plt.show()

    logger.info(
        f"Seuil optimal : {optimal_threshold:.2f} -> "
        f"F1 = {f1_scores[optimal_idx]:.4f}"
    )
    return optimal_threshold
