# Projet MLOps : Détection de Fraude Bancaire

## Objectif
Détecter les fraudes sur les transactions bancaires en utilisant un pipeline MLOps complet.

## Installation
```bash
pip install -r requirements.txt

python src/train.py
#VISUALISATION:
mlflow ui

TEST : 
pytest

PREDECTION
python src/predict.py


1. `python src/data_analysis.py` → explorer les données  
2. `python src/train.py` → entraîner les modèles + GridSearch + MLflow  
3. `python src/evaluate.py` → générer le rapport de drift  
4. `python src/predict.py` → prédictions CSV  
5. `mlflow ui` → visualiser les runs  
