import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import mlflow
import mlflow.sklearn
import joblib

from data import load_data, preprocess_data

# Charger et préparer les données
df = load_data(r"C:\Users\amrga\OneDrive\Desktop\Fraud-mlops\data\creditcard.csv")
df = preprocess_data(df)

#Echantillonage 
df = df.sample(n=5000, random_state=42)

# Préprocessing
df = preprocess_data(df)

X = df.drop('Class', axis=1)
y = df['Class']

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# SMOTE sur train seulement
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

mlflow.set_experiment("Fraud Detection")

# ---------------- Logistic Regression ----------------
with mlflow.start_run(run_name="LogisticRegression"):
    lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    lr.fit(X_train_res, y_train_res)
    y_pred = lr.predict(X_test)

    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, lr.predict_proba(X_test)[:,1])

    print(f"📊 LogisticRegression → Recall: {recall:.4f}, Precision: {precision:.4f}")

    mlflow.log_params({'model':'LogisticRegression'})
    mlflow.log_metrics({'recall':recall, 'precision':precision, 'f1':f1, 'roc_auc':roc})
    mlflow.sklearn.log_model(lr, "FraudDetector_v1_LR")

# ---------------- Random Forest + GridSearch ----------------
with mlflow.start_run(run_name="RandomForest"):
    rf = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    grid = GridSearchCV(rf, param_grid, scoring='recall', cv=3, n_jobs=-1)
    grid.fit(X_train_res, y_train_res)

    best_rf = grid.best_estimator_
    y_pred = best_rf.predict(X_test)

    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, best_rf.predict_proba(X_test)[:,1])

    print(f"📊 RandomForest → Recall: {recall:.4f}, Precision: {precision:.4f}")
    print("Best RF params:", grid.best_params_)

    mlflow.log_params(grid.best_params_)
    mlflow.log_metrics({'recall':recall, 'precision':precision, 'f1':f1, 'roc_auc':roc})
    mlflow.sklearn.log_model(best_rf, "FraudDetector_v1_RF")

# Sauvegarder le meilleur modèle (ici RandomForest)
joblib.dump(best_rf, r"C:\Users\amrga\OneDrive\Desktop\Fraud-mlops\model\model.pkl")
print("✅ Meilleur modèle sauvegardé !")