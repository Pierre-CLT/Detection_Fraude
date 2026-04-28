# 🔍 Détection de Fraude sur Transactions Bancaires

Projet complet de Machine Learning pour détecter les transactions frauduleuses dans un dataset de 284 807 transactions bancaires. De l'exploration des données au déploiement d'un modèle interprétable.

## 📊 Résultats clés

| Métrique | Score |
|----------|-------|
| **Modèle retenu** | XGBoost + Pondération |
| **F1-Score** | 0.860 |
| **Precision** | 0.874 |
| **Recall** | 0.847 |
| **AUC-ROC** | 0.975 |
| **Fraudes détectées** | 83/98 sur le jeu de test |

Le modèle détecte **85% des fraudes** avec seulement **12 fausses alertes** sur 56 962 transactions.

## 🗂️ Structure du projet

```
Detection_Fraude/
├── data/                          # Dataset (non versionné)
│   └── creditcard.csv
├── notebooks/
│   ├── 01_exploration.ipynb       # EDA et visualisations
│   ├── 02_preprocessing.ipynb     # Normalisation et feature engineering
│   ├── 03_modelisation_supervisee.ipynb  # 12 combinaisons modèle/stratégie
│   ├── 04_modelisation_non_supervisee.ipynb  # Isolation Forest et Autoencoder
│   └── 05_evaluation_interpretabilite.ipynb  # SHAP et comparaison finale
├── outputs/                       # Graphiques et modèles sauvegardés
├── src/                           # Fonctions utilitaires
├── .gitignore
├── requirements.txt
└── README.md
```

## 🔬 Méthodologie

### Étape 1 — Exploration des données
- Dataset de 284 807 transactions dont **492 fraudes (0.17%)**
- 28 features anonymisées par PCA (V1-V28) + Amount + Time
- Identification des features discriminantes : V14, V17, V12 présentent la meilleure séparation entre fraudes et transactions légitimes

### Étape 2 — Prétraitement & Feature Engineering
- Normalisation de Amount et Time avec StandardScaler
- Création de 4 nouvelles features :
  - `Hour` : heure de la journée (pic de fraude détecté entre 2h-4h)
  - `Is_Night` : indicateur binaire nuit/jour
  - `Amount_log` : log du montant (réduction des valeurs extrêmes)
  - `Amount_category` : catégorisation par tranche de montant
- Split train/test stratifié (80/20) pour conserver le ratio de fraude

### Étape 3 — Gestion du déséquilibre de classes
Quatre stratégies testées :
- **Original** : données brutes (0.17% de fraudes)
- **SMOTE** : sur-échantillonnage synthétique par interpolation
- **Under-sampling** : sous-échantillonnage de la classe majoritaire
- **Pondération** : ajustement des poids dans la fonction de coût

### Étape 4 — Modélisation supervisée
12 combinaisons testées (3 modèles × 4 stratégies) :
- Logistic Regression (baseline)
- Random Forest (ensemble)
- XGBoost (gradient boosting) ← **meilleur modèle**

### Étape 5 — Modélisation non supervisée
- **Isolation Forest** : F1 = 0.237 — détecte les anomalies mais manque de précision
- **Autoencoder** : F1 = 0.483 — meilleur que Isolation Forest mais inférieur au supervisé
- Ces approches restent complémentaires en production pour détecter des fraudes inédites

### Étape 6 — Interprétabilité (SHAP)
- V14 est la feature la plus importante (impact SHAP moyen de 3.0)
- Les résultats SHAP confirment les observations de l'EDA
- Explication transaction par transaction pour justifier chaque alerte

## ⚙️ Technologies utilisées

- **Python 3.12**
- **Analyse** : Pandas, NumPy
- **Visualisation** : Matplotlib, Seaborn
- **Machine Learning** : Scikit-learn, XGBoost, Imbalanced-learn
- **Deep Learning** : TensorFlow/Keras (Autoencoder)
- **Interprétabilité** : SHAP

## 🚀 Reproduire le projet

```bash
# 1. Cloner le repo
git clone https://github.com/Pierre-CLT/Detection_Fraude.git
cd Detection_Fraude

# 2. Créer l'environnement virtuel
python -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\Activate.ps1     # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Télécharger le dataset
# → https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# Placer creditcard.csv dans le dossier data/

# 5. Exécuter les notebooks dans l'ordre (01 → 05)
```

## 📈 Ce que j'ai appris

- **Gestion du déséquilibre** : l'accuracy est trompeuse sur des données déséquilibrées. Le F1-Score et l'AUC-ROC sont les métriques pertinentes.
- **Compromis Précision/Recall** : en anti-fraude, une fraude non détectée (FN) coûte plus cher qu'une fausse alerte (FP). Le Recall est prioritaire, mais un bon F1 garantit l'équilibre.
- **Supervisé vs Non supervisé** : le supervisé domine quand les labels existent, mais le non supervisé détecte les anomalies inédites. Les deux sont complémentaires.
- **Interprétabilité** : SHAP transforme une boîte noire en décision explicable — indispensable dans le secteur bancaire.
- **Feature engineering** : transformer Time en heure de la journée a créé une feature exploitée par le modèle (visible dans SHAP).

## 📝 Dataset

[Credit Card Fraud Detection — Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
> Dataset publié par le Machine Learning Group de l'Université Libre de Bruxelles (ULB).
> 284 807 transactions sur 2 jours, dont 492 fraudes. Features V1-V28 anonymisées par PCA.