import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from scipy.stats import mannwhitneyu

# Backend non interactif
plt.switch_backend('Agg')

# ==============================
# 📁 Paths (CORRIGÉ)
# ==============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "creditcard.csv")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

os.makedirs(PLOTS_DIR, exist_ok=True)

# Debug (optionnel)
print("BASE_DIR:", BASE_DIR)
print("DATA_PATH:", DATA_PATH)

# ==============================
# 📥 Vérification dataset
# ==============================
if not os.path.exists(DATA_PATH):
    print(f"❌ Error: Data file not found at {DATA_PATH}")
    exit(1)

# ==============================
# 📊 Chargement
# ==============================
df = pd.read_csv(DATA_PATH)
print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ==============================
# 1️⃣ Distribution Amount (log)
# ==============================
plt.figure(figsize=(8,5))
sns.histplot(np.log1p(df['Amount']), bins=50, kde=True)
plt.title("Distribution log(Amount)")
plt.xlabel("log(Amount)")

plot_path = os.path.join(PLOTS_DIR, "01_amount_distribution.png")
plt.savefig(plot_path, dpi=100, bbox_inches='tight')
plt.close()

print(f"📊 Saved: {plot_path}")

# ==============================
# 2️⃣ Amount by Class
# ==============================
plt.figure(figsize=(8,5))
sns.histplot(data=df, x=np.log1p(df['Amount']), hue='Class', bins=50, kde=True)
plt.title("Distribution log(Amount) par classe")
plt.xlabel("log(Amount)")

plot_path = os.path.join(PLOTS_DIR, "02_amount_by_class.png")
plt.savefig(plot_path, dpi=100, bbox_inches='tight')
plt.close()

print(f"📊 Saved: {plot_path}")

# ==============================
# 3️⃣ Corrélation avec Class
# ==============================
features = [f'V{i}' for i in range(1, 29)]

corr = df[features + ['Class']].corr()

plt.figure(figsize=(8,10))
corr_sorted = corr[['Class']].sort_values(by='Class', ascending=False)

sns.heatmap(corr_sorted, cmap='coolwarm', annot=False)
plt.title("Corrélation des variables avec Class")

plot_path = os.path.join(PLOTS_DIR, "03_correlation_class.png")
plt.savefig(plot_path, dpi=100, bbox_inches='tight')
plt.close()

print(f"📊 Saved: {plot_path}")

# ==============================
# 4️⃣ Mann-Whitney U Test
# ==============================
print("\n📈 Mann-Whitney U Test Results (trié):")

results = []

for f in features:
    stat, p = mannwhitneyu(
        df[df['Class'] == 0][f],
        df[df['Class'] == 1][f]
    )
    results.append((f, p))

# Trier par p-value
results = sorted(results, key=lambda x: x[1])

for f, p in results:
    print(f"{f}: p-value={p:.6f}")

print("\n✅ Data analysis completed successfully!")