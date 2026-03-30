import pandas as pd
import joblib

# Charger modèle
model = joblib.load(r"C:\Users\amrga\OneDrive\Desktop\Fraud-mlops\model\model.pkl")

# Lire le CSV de test
df = pd.read_csv(r"C:\Users\amrga\OneDrive\Desktop\Fraud-mlops\data\creditcard_test.csv")

# Séparer features
X = df.drop('Class', axis=1, errors='ignore')

# Prédictions
df['FraudScore'] = model.predict_proba(X)[:,1]

# Sauvegarder résultats
df.to_csv(r"C:\Users\amrga\OneDrive\Desktop\Fraud-mlops\data\predictions.csv", index=False)
print("✅ Prédictions sauvegardées !")