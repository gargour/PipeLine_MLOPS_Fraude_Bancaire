import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# Dataset référence
df_ref = pd.read_csv(r"C:\Users\amrga\OneDrive\Desktop\Fraud-mlops\data\creditcard.csv")

# Dataset actuel (simulate drift)
df_cur = df_ref.copy()
df_cur['Amount'] *= 1.5  # augmenter le montant pour simuler dérive

# Générer rapport Evidently
report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=df_ref, current_data=df_cur)
report.save_html(r"C:\Users\amrga\OneDrive\Desktop\Fraud-mlops\reports\drift_report.html")

print("✅ Rapport de dérive généré !")