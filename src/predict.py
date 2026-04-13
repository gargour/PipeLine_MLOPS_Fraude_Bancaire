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
# =========================
# 📊 GRAPH 3: BATCH FRAUD ANALYSIS (Koll 1000)
# =========================
batch_size = 1000
# N-rattbu el-data mel-score el-3ali lel-score el-waati
df_sorted = df.sort_values(by="FraudScore", ascending=False).reset_index(drop=True)

# Na3mlou 9asmet el-batches
df_sorted['Batch'] = df_sorted.index // batch_size

# Na7sbu el-fraudes el-7a9iqyia (Class) fi koll batch
batch_counts = df_sorted.groupby('Batch')['Class'].sum().reset_index()

# N-sawrou el-10 batches el-wala (Top 10,000 transactions)
plt.figure(figsize=(12,6))
plt.bar(batch_counts['Batch'][:10], batch_counts['Class'][:10], color='salmon', edgecolor='black')

plt.title(f"Nombre de Fraudes Réelles détectées par tranches de {batch_size} transactions")
plt.xlabel(f"Numéro du Batch (Chaque batch = {batch_size} transactions)")
plt.ylabel("Nombre de Fraudes Détectées")
plt.xticks(range(10)) # N-warriw el-10 batches el-wala
plt.grid(axis='y', linestyle='--', alpha=0.7)

# N-zidu el-numruat fou9 el-barrat bech t-koun awda7
for i, v in enumerate(batch_counts['Class'][:10]):
    plt.text(i, v + 0.5, str(int(v)), ha='center', fontweight='bold')

plt.show()