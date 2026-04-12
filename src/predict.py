import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

# =========================
# 📁 PATHS
# =========================
BASE_DIR = r"C:\Users\amrga\OneDrive\Desktop\Fraud-mlops"

MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "creditcard.csv")  # ✔ FIX IMPORTANT
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "predictions.csv")

# =========================
# 🤖 LOAD MODEL
# =========================
model = joblib.load(MODEL_PATH)

# =========================
# 📊 LOAD DATA
# =========================
df = pd.read_csv(DATA_PATH)

# =========================
# 🔹 FEATURES
# =========================
if 'Class' in df.columns:
    X = df.drop('Class', axis=1)
else:
    X = df.copy()

# =========================
# 🔥 PREDICTION
# =========================
df['Prediction'] = model.predict(X)

# ✔ safe probability check
if hasattr(model, "predict_proba"):
    df['FraudScore'] = model.predict_proba(X)[:, 1]
else:
    df['FraudScore'] = 0

# =========================
# 🚨 FRAUD FILTER
# =========================
fraudes = df[df['Prediction'] == 1]

print("===================================")
print("🚨 Nombre de fraudes détectées:", len(fraudes))
print("===================================")

# =========================
# 💾 SAVE RESULTS
# =========================
df.to_csv(OUTPUT_PATH, index=False)
print("✅ Predictions sauvegardées dans:", OUTPUT_PATH)

# =========================
# 📊 GRAPH 1: SCORE CURVE
# =========================
df_sorted = df.sort_values(by="FraudScore")

plt.figure(figsize=(12,5))
plt.plot(df_sorted["FraudScore"].values, label="Fraud Score")

plt.axhline(y=0.8, color='r', linestyle='--', label="Threshold 0.8")

plt.title("Fraud Detection Score Curve")
plt.xlabel("Transactions")
plt.ylabel("Fraud Probability")
plt.legend()
plt.show()
# =========================
# 📊 GRAPH 2: DISTRIBUTION (CORRECTED)
# =========================
plt.figure(figsize=(12,5))

# Nesta3mlou log=True bech el-fraude (elli 3dadhom sghir) ybenou m3a el-normal (elli 3dadhom kbir)
plt.hist(df[df["Class"] == 0]["FraudScore"], 
         bins=50, alpha=0.5, label="Normal", color='skyblue', log=True)

plt.hist(df[df["Class"] == 1]["FraudScore"], 
         bins=50, alpha=0.7, label="Fraud", color='orange', log=True)

plt.title("Fraud vs Normal Distribution (Log Scale)")
plt.xlabel("Fraud Score")
plt.ylabel("Frequency (Log Scale)") # Matensech tbeddel el-label khater el-echelle tbedlet
plt.legend()
plt.grid(axis='y', alpha=0.3) # Zid grid bech t-sahal el-9raya
plt.show()