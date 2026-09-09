# Prédiction du Churn Client - Machine Learning
### Modélisation du risque de résiliation client - Telco Customer Dataset

**Samir NZAMBA** - Mastère DPIA 1 - Fonderie de l'Image - Paris - 2025-2026  
Projet #3 - Introduction au Machine Learning

---

## À propos

Ce projet construit un pipeline complet de Machine Learning pour prédire le churn des clients d'un opérateur télécom. L'objectif est d'anticiper les résiliations avant qu'elles ne surviennent, afin de permettre des actions de rétention ciblées.

Le pipeline couvre l'intégralité de la chaîne : exploration et nettoyage des données, encodage et rééquilibrage des classes via SMOTE, entraînement et comparaison de 5 algorithmes (Régression Logistique, Arbre de Décision, Random Forest, XGBoost, LightGBM), optimisation par GridSearchCV, validation croisée 5-fold, explicabilité individuelle via SHAP, monitoring du data drift avec Evidently et exposition du modèle via une API REST FastAPI. Le tout est déployé dans un dashboard Streamlit interactif accessible en ligne.

---

## Stack technique

**Machine Learning** : scikit-learn, XGBoost, LightGBM, SHAP  
**Données** : pandas, numpy, imbalanced-learn (SMOTE)  
**Visualisations** : matplotlib, seaborn, plotly  
**Dashboard** : Streamlit (déployé sur Streamlit Cloud)  
**API REST** : FastAPI, uvicorn  
**Monitoring** : Evidently (data drift + performance)  
**Tests** : pytest (6 tests unitaires)  
**CI/CD** : GitHub Actions (flake8 + pytest)  
**Versionnement** : Git, GitHub

---

## Architecture

```
Dataset Telco Customer Churn (Kaggle)
                |
                v
         src/preprocessing.py
    (nettoyage - encodage - SMOTE - split)
                |
                v
         src/models.py
  (5 algorithmes - GridSearchCV - cross-val)
                |
                v
         src/evaluation.py
   (métriques - ROC - seuil optimal - SHAP)
                |
         +------+------+
         v             v
   dashboard/       api/
   app.py           main.py
   (Streamlit)      (FastAPI)
         |
         v
  Streamlit Cloud
  (déploiement en ligne)
```

---

## Résultats

**Modèle retenu : XGBoost Optimisé**

| Métrique | Valeur |
|---|---|
| AUC-ROC | 0.8420 |
| Accuracy | 76 % |
| Recall (Churné) | 79 % |
| F1-score | 0.64 |
| Seuil de décision | 0.10 (optimisé) |

Sur 374 clients ayant réellement churné dans le jeu de test, le modèle en détecte **295** (79 %).

**Top 5 facteurs de churn (SHAP) :**

| Rang | Variable | Importance SHAP |
|---|---|---|
| 1 | Contract_Two year | 0.813 |
| 2 | tenure | 0.656 |
| 3 | Contract_One year | 0.414 |
| 4 | InternetService_Fiber optic | 0.392 |
| 5 | MonthlyCharges | 0.350 |

---

## Structure du projet

```
churn_project/
+-- .github/
|   +-- workflows/
|       +-- ci.yml                          # Pipeline CI/CD GitHub Actions
+-- api/
|   +-- main.py                             # API REST FastAPI
+-- dashboard/
|   +-- app.py                              # Dashboard Streamlit (5 pages)
+-- data/                                   # Données brutes (non versionnées)
|   +-- WA_Fn-UseC_-Telco-Customer-Churn.csv
+-- exports/                                # Fichiers générés par les notebooks
|   +-- dataset_clean.csv
|   +-- resultats_modeles.csv
|   +-- validation_croisee.csv
|   +-- shap_importance.csv
|   +-- evidently_report.html
|   +-- 01_distribution_churn.png
|   +-- ...
|   +-- 15_shap_dependence.png
+-- models/                                 # Modèles sérialisés
|   +-- scaler.pkl
|   +-- feature_names.pkl
|   +-- optimal_threshold.pkl
|   +-- shap_explainer.pkl
|   +-- xgboost_best.pkl
|   +-- ...
+-- notebooks/
|   +-- 01_eda_preparation.ipynb            # Exploration et préparation des données
|   +-- 02_modelisation_validation.ipynb    # Modélisation, optimisation, SHAP, Evidently
+-- src/                                    # Modules Python réutilisables
|   +-- __init__.py
|   +-- preprocessing.py                    # Nettoyage, encodage, SMOTE, split/scale
|   +-- models.py                           # Définition, entraînement, optimisation
|   +-- evaluation.py                       # Métriques, visualisations, seuil optimal
+-- tests/
|   +-- __init__.py
|   +-- test_preprocessing.py               # 6 tests unitaires
+-- .gitignore
+-- requirements.txt
+-- README.md
```

---

## Installation locale

### Prérequis

- Python 3.11 ou supérieur
- Git

### 1. Cloner le dépôt

```bash
git clone https://github.com/SNZAMBA65/churn_project.git
cd churn_project
```

### 2. Créer l'environnement virtuel

```bash
python -m venv venv
source venv/Scripts/activate    # Windows
# source venv/bin/activate      # Linux / macOS
pip install -r requirements.txt
```

### 3. Télécharger le dataset

```bash
# Via l'API Kaggle
kaggle datasets download -d blastchar/telco-customer-churn -p data/ --unzip

# Ou manuellement depuis :
# https://www.kaggle.com/datasets/blastchar/telco-customer-churn
# Placer le CSV dans data/
```

### 4. Exécuter les notebooks

```bash
jupyter notebook
```

Exécuter dans l'ordre :

1. `notebooks/01_eda_preparation.ipynb` - Exploration et préparation des données
2. `notebooks/02_modelisation_validation.ipynb` - Modélisation, SHAP, Evidently

### 5. Lancer le dashboard

```bash
cd dashboard
streamlit run app.py
```

Ouvrir **http://localhost:8501**

### 6. Lancer l'API REST

```bash
uvicorn api.main:app --reload
```

Documentation interactive disponible sur **http://localhost:8000/docs**

---

## Dashboard Streamlit

Le dashboard interactif comprend 5 pages :

| Page | Contenu |
|---|---|
| Vue d'ensemble | KPIs clés, résumé exécutif, facteurs de risque majeurs |
| Analyse exploratoire | Visualisations EDA interactives par thème |
| Performance des modèles | Courbes ROC, matrices de confusion, feature importance, validation croisée |
| Qualité & Monitoring | Seuil de décision optimisé, learning curve, rapport Evidently |
| Simulateur | Prédiction temps réel avec détection des risques et recommandations |

Compatible light/dark mode via les variables CSS natives de Streamlit.

---

## API REST

L'API FastAPI expose 4 endpoints :

| Méthode | Route | Description |
|---|---|---|
| GET | `/` | Statut de l'API |
| GET | `/health` | Santé du service et informations du modèle |
| POST | `/predict` | Prédiction individuelle avec explicabilité SHAP |
| POST | `/predict/batch` | Prédiction en lot (max 100 clients) |

### Exemple de requête

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Male",
    "SeniorCitizen": "No",
    "Partner": "No",
    "Dependents": "No",
    "tenure": 3,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 95.0,
    "TotalCharges": 285.0
  }'
```

### Exemple de réponse

```json
{
  "probabilite_churn": 0.8823,
  "probabilite_pct": "88.2 %",
  "prediction": "Churner",
  "niveau_risque": "ÉLEVÉ",
  "seuil_utilise": 0.1,
  "facteurs_principaux": [
    {
      "feature": "Contract_Two year",
      "impact": 0.4137,
      "direction": "vers le churn"
    },
    {
      "feature": "InternetService_Fiber optic",
      "impact": 0.4066,
      "direction": "vers le churn"
    }
  ]
}
```

---

## Pipeline src/

Les modules `src/` centralisent toute la logique métier, importée par les notebooks, le dashboard et l'API.

```python
src/preprocessing.py
    load_raw_data()              # Chargement du dataset brut
    clean_data()                 # Nettoyage des anomalies
    encode_features()            # Label Encoding + One-Hot Encoding
    split_and_scale()            # Split train/test + StandardScaler + SMOTE
    run_preprocessing_pipeline() # Pipeline complet en une commande

src/models.py
    get_models()                 # Dictionnaire des 5 modèles
    get_param_grids()            # Grilles GridSearchCV
    train_all_models()           # Entrainement et sauvegarde
    optimize_model()             # GridSearchCV sur un modèle
    cross_validate_models()      # Validation croisée 5-fold

src/evaluation.py
    evaluate_model()             # Métriques pour un modèle
    evaluate_all_models()        # Comparaison de tous les modèles
    plot_roc_curves()            # Courbes ROC comparatives
    plot_confusion_matrices()    # Matrices de confusion côte à côte
    plot_learning_curve()        # Détection overfitting/underfitting
    find_optimal_threshold()     # Optimisation du seuil de décision
```

---

## Tests

```bash
pytest tests/ -v
```

6 tests unitaires couvrant le module `preprocessing.py` :

| Test | Description |
|---|---|
| `test_clean_data_removes_customer_id` | Suppression de l'identifiant client |
| `test_clean_data_fixes_total_charges` | Conversion TotalCharges en float |
| `test_clean_data_handles_empty_total_charges` | Valeurs vides remplacées par 0 |
| `test_clean_data_converts_senior_citizen` | Harmonisation 0/1 vers No/Yes |
| `test_encode_features_produces_numeric_columns` | Encodage complet sans colonne texte |
| `test_encode_features_keeps_same_row_count` | Aucune ligne perdue à l'encodage |

---

## CI/CD

Le pipeline GitHub Actions se déclenche à chaque push sur `main` :

1. Configuration Python 3.11
2. Installation des dépendances (`requirements.txt`)
3. Vérification du style de code (`flake8` - PEP 8)
4. Vérification des imports `src/`
5. Exécution des 6 tests unitaires (`pytest`)

---

## Compétences démontrées

**D-02 - Algorithme d'IA adapté aux données**
- 5 algorithmes implémentés, comparés et justifiés
- Gestion du déséquilibre des classes (SMOTE)
- Optimisation des hyperparamètres (GridSearchCV)
- Seuil de décision adapté au contexte métier
- Explicabilité individuelle via SHAP

**D-04 - Pipeline CI/CD**
- GitHub Actions configuré (lint, tests, vérification imports)
- 6 tests unitaires sur le module de prétraitement
- Commits conventionnels en français

**D-06 - Monitoring de la solution d'IA**
- Rapport Evidently (data drift + performance de classification)
- Courbe d'apprentissage (détection overfitting/underfitting)
- Validation croisée 5-fold pour évaluer la stabilité

---

## Liens

- **Dashboard en ligne** : https://nzb-churn-project.streamlit.app/
- **GitHub** : https://github.com/SNZAMBA65/churn_project
- **API locale** : http://localhost:8000/docs
- **Dataset** : https://www.kaggle.com/datasets/blastchar/telco-customer-churn

---

## Contexte académique

Projet réalisé dans le cadre du Mastère DPIA 1 - Directeur de Projet en Intelligence Artificielle à Fonderie de l'Image (Paris), année 2025-2026.

Dataset source : [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) - IBM Sample Data (Kaggle).