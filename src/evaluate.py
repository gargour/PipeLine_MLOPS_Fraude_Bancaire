import pandas as pd
from evidently.report import Report
from evidently.presets import DataDriftPreset
import os

# 📁 Chemins
BASE_DIR = r"C:\Users\amrga\OneDrive\Desktop\Fraud-mlops"
DATA_PATH = os.path.join(BASE_DIR, "data", "creditcard.csv")
REPORT_PATH = os.path.join(BASE_DIR, "reports", "drift_report.html")

# 📊 Charger les données
df_ref = pd.read_csv(DATA_PATH)

# 📊 Simuler un dataset avec dérive
df_cur = df_ref.copy()
df_cur['Amount'] = df_cur['Amount'] * 1.5  # simulation drift

# 📈 Créer le rapport de dérive
report = Report(metrics=[DataDriftPreset()])

report.run(
    reference_data=df_ref,
    current_data=df_cur
)

# 📁 Créer le dossier reports s’il n’existe pas
os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)

# 💾 Générer et sauvegarder le HTML
html_content = report.as_html()
with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(html_content)

print("✅ Rapport de dérive généré avec succès !")
print(f"📄 Chemin : {REPORT_PATH}")