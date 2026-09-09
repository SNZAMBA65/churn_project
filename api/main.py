# ============================================================
# api/main.py
# API REST: Prédiction du Churn Client
# Projet #3 - DPIA 1 - Samir NZAMBA
# ============================================================
# Usage : uvicorn api.main:app --reload
# Docs  : http://localhost:8000/docs
# ============================================================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
import os
import shap

# ─── Chargement des modèles ───────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

model = joblib.load(os.path.join(MODELS_DIR, "xgboost_best.pkl"))
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))
optimal_threshold = joblib.load(os.path.join(MODELS_DIR, "optimal_threshold.pkl"))
explainer = joblib.load(os.path.join(MODELS_DIR, "shap_explainer.pkl"))

# ─── Application FastAPI ──────────────────────────────────
app = FastAPI(
    title="Churn Prediction API",
    description="API de prédiction du churn client - Projet #3 DPIA 1",
    version="1.0.0",
)

# ─── Schéma de la requête ─────────────────────────────────
class ClientProfile(BaseModel):
    gender: str = Field(..., example="Male")
    SeniorCitizen: str = Field(..., example="No")
    Partner: str = Field(..., example="Yes")
    Dependents: str = Field(..., example="No")
    tenure: int = Field(..., ge=0, le=150, example=12)
    PhoneService: str = Field(..., example="Yes")
    MultipleLines: str = Field(..., example="No")
    InternetService: str = Field(..., example="Fiber optic")
    OnlineSecurity: str = Field(..., example="No")
    OnlineBackup: str = Field(..., example="No")
    DeviceProtection: str = Field(..., example="No")
    TechSupport: str = Field(..., example="No")
    StreamingTV: str = Field(..., example="No")
    StreamingMovies: str = Field(..., example="No")
    Contract: str = Field(..., example="Month-to-month")
    PaperlessBilling: str = Field(..., example="Yes")
    PaymentMethod: str = Field(..., example="Electronic check")
    MonthlyCharges: float = Field(..., ge=0, le=250, example=70.0)
    TotalCharges: float = Field(..., ge=0, le=15000, example=840.0)


# ─── Encodage du profil client ────────────────────────────
def encode_client(profile: ClientProfile) -> pd.DataFrame:
    binary_map = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
    client = {
        "gender": binary_map.get(profile.gender, 0),
        "SeniorCitizen": binary_map.get(profile.SeniorCitizen, 0),
        "Partner": binary_map.get(profile.Partner, 0),
        "Dependents": binary_map.get(profile.Dependents, 0),
        "tenure": profile.tenure,
        "PhoneService": binary_map.get(profile.PhoneService, 0),
        "PaperlessBilling": binary_map.get(profile.PaperlessBilling, 0),
        "MonthlyCharges": profile.MonthlyCharges,
        "TotalCharges": profile.TotalCharges,
        "MultipleLines_No phone service": 1 if profile.MultipleLines == "No phone service" else 0,
        "MultipleLines_Yes": 1 if profile.MultipleLines == "Yes" else 0,
        "InternetService_Fiber optic": 1 if profile.InternetService == "Fiber optic" else 0,
        "InternetService_No": 1 if profile.InternetService == "No" else 0,
        "OnlineSecurity_No internet service": 1 if profile.OnlineSecurity == "No internet service" else 0,
        "OnlineSecurity_Yes": 1 if profile.OnlineSecurity == "Yes" else 0,
        "OnlineBackup_No internet service": 1 if profile.OnlineBackup == "No internet service" else 0,
        "OnlineBackup_Yes": 1 if profile.OnlineBackup == "Yes" else 0,
        "DeviceProtection_No internet service": 1 if profile.DeviceProtection == "No internet service" else 0,
        "DeviceProtection_Yes": 1 if profile.DeviceProtection == "Yes" else 0,
        "TechSupport_No internet service": 1 if profile.TechSupport == "No internet service" else 0,
        "TechSupport_Yes": 1 if profile.TechSupport == "Yes" else 0,
        "StreamingTV_No internet service": 1 if profile.StreamingTV == "No internet service" else 0,
        "StreamingTV_Yes": 1 if profile.StreamingTV == "Yes" else 0,
        "StreamingMovies_No internet service": 1 if profile.StreamingMovies == "No internet service" else 0,
        "StreamingMovies_Yes": 1 if profile.StreamingMovies == "Yes" else 0,
        "Contract_One year": 1 if profile.Contract == "One year" else 0,
        "Contract_Two year": 1 if profile.Contract == "Two year" else 0,
        "PaymentMethod_Credit card (automatic)": 1 if profile.PaymentMethod == "Credit card (automatic)" else 0,
        "PaymentMethod_Electronic check": 1 if profile.PaymentMethod == "Electronic check" else 0,
        "PaymentMethod_Mailed check": 1 if profile.PaymentMethod == "Mailed check" else 0,
    }
    return pd.DataFrame([client])[feature_names]


# ─── Routes ───────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "message": "Churn Prediction API — opérationnelle",
        "version": "1.0.0",
        "endpoints": {
            "prediction": "/predict",
            "documentation": "/docs",
            "sante": "/health"
        }
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "modele": "XGBoost Optimisé",
        "seuil_optimal": round(float(optimal_threshold), 2),
        "features": len(feature_names)
    }


@app.post("/predict")
def predict(profile: ClientProfile):
    try:
        # Encodage et normalisation
        X = encode_client(profile)
        X_scaled = scaler.transform(X)

        # Prédiction
        proba = float(model.predict_proba(X_scaled)[0][1])
        prediction = int(proba >= optimal_threshold)

        # Niveau de risque
        if proba < 0.35:
            risk_level = "FAIBLE"
        elif proba < 0.60:
            risk_level = "MODÉRÉ"
        else:
            risk_level = "ÉLEVÉ"

        # Explicabilité SHAP
        shap_values = explainer.shap_values(pd.DataFrame(X_scaled, columns=feature_names))
        shap_top = pd.DataFrame({
            "feature": feature_names,
            "shap_value": shap_values[0]
        }).sort_values("shap_value", key=abs, ascending=False).head(5)

        facteurs_principaux = [
            {
                "feature": row["feature"],
                "impact": round(float(row["shap_value"]), 4),
                "direction": "vers le churn" if row["shap_value"] > 0 else "protège du churn"
            }
            for _, row in shap_top.iterrows()
        ]

        return {
            "probabilite_churn": round(proba, 4),
            "probabilite_pct": f"{proba*100:.1f} %",
            "prediction": "Churner" if prediction else "Stable",
            "niveau_risque": risk_level,
            "seuil_utilise": round(float(optimal_threshold), 2),
            "facteurs_principaux": facteurs_principaux,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
def predict_batch(profiles: list[ClientProfile]):
    """Prédiction en lot pour plusieurs clients simultanément."""
    if len(profiles) > 100:
        raise HTTPException(
            status_code=400,
            detail="Maximum 100 clients par requête batch"
        )
    results = []
    for i, profile in enumerate(profiles):
        try:
            X = encode_client(profile)
            X_scaled = scaler.transform(X)
            proba = float(model.predict_proba(X_scaled)[0][1])
            prediction = int(proba >= optimal_threshold)
            results.append({
                "index": i,
                "probabilite_churn": round(proba, 4),
                "prediction": "Churner" if prediction else "Stable",
                "niveau_risque": "ÉLEVÉ" if proba >= 0.6 else "MODÉRÉ" if proba >= 0.35 else "FAIBLE"
            })
        except Exception as e:
            results.append({"index": i, "erreur": str(e)})
    return {"total": len(profiles), "resultats": results}