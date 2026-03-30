import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess_data(df):
    df = df.copy()  # 🔥 éviter SettingWithCopyWarning

    # Supprimer les doublons
    df = df.drop_duplicates()

    # Remplacer les valeurs manquantes
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].fillna(df[col].mean())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])

    # Normaliser Time et Amount
    scaler = StandardScaler()
    df.loc[:, ['Time', 'Amount']] = scaler.fit_transform(df[['Time', 'Amount']])

    # Vérifier type Class
    df.loc[:, 'Class'] = df['Class'].astype(int)

    return df