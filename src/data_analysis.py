import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu

df = pd.read_csv(r"C:\Users\amrga\OneDrive\Desktop\Fraud-mlops\data\creditcard.csv")

# Histogramme Amount
plt.figure(figsize=(8,5))
sns.histplot(df['Amount'], bins=50, kde=True)
plt.title("Distribution Amount")
plt.show()

# Distribution par classe
plt.figure(figsize=(8,5))
sns.histplot(data=df, x='Amount', hue='Class', bins=50, kde=True, multiple='stack')
plt.title("Distribution Amount par classe")
plt.show()

# Heatmap corrélation V1-V28
plt.figure(figsize=(12,8))
corr = df[[f'V{i}' for i in range(1,29)] + ['Class']].corr()
sns.heatmap(corr, cmap='coolwarm')
plt.title("Corrélation V1-V28 vs Class")
plt.show()

# Test Mann-Whitney
features = [f'V{i}' for i in range(1,29)]
for f in features:
    stat, p = mannwhitneyu(df[df['Class']==0][f], df[df['Class']==1][f])
    print(f"{f}: p-value={p:.5f}")