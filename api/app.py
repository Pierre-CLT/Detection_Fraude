# ============================================
# API de Détection de Fraude — FastAPI
# ============================================
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
import shap

# Chargement du modèle
with open("model.pkl", "rb") as f:
    package = pickle.load(f)

model = package["model"]
feature_names = package["feature_names"]
expected_value = package["expected_value"]
explainer = shap.TreeExplainer(model)

# Initialisation de l'API
app = FastAPI(
    title="API Détection de Fraude",
    description="Prédit si une transaction bancaire est frauduleuse",
    version="1.0"
)

# Autoriser les requêtes depuis le frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Schéma de la requête
class Transaction(BaseModel):
    features: list[float]


# Endpoint de santé
@app.get("/")
def root():
    return {
        "status": "online",
        "model": "XGBoost + Pondération",
        "features_expected": len(feature_names)
    }


# Endpoint de prédiction
@app.post("/predict")
def predict(transaction: Transaction):
    # Vérification du nombre de features
    if len(transaction.features) != len(feature_names):
        return {
            "error": f"Attendu {len(feature_names)} features, reçu {len(transaction.features)}"
        }

    # Prédiction
    X = pd.DataFrame([transaction.features], columns=feature_names)
    proba = model.predict_proba(X)[:, 1][0]
    prediction = "FRAUDE" if proba >= 0.5 else "LÉGITIME"

    # Explications SHAP (top 5 features)
    shap_values = explainer.shap_values(X)[0]
    top_idx = np.argsort(np.abs(shap_values))[-5:][::-1]
    explanations = [
        {
            "feature": feature_names[i],
            "value": round(float(X.iloc[0, i]), 4),
            "shap_impact": round(float(shap_values[i]), 4),
            "direction": "fraude" if shap_values[i] > 0 else "légitime"
        }
        for i in top_idx
    ]

    return {
        "prediction": prediction,
        "probability": round(float(proba), 4),
        "threshold": 0.5,
        "top_factors": explanations
    }