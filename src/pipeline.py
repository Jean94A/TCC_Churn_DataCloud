import pandas as pd
import numpy as np
import os
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def ejecutar_pipeline():
    print("🚀 [1/4] Cargando datos...")
    df = pd.read_csv('../data/datacloud_churn.csv')

    print("🧹 [2/4] Limpiando datos y realizando Feature Engineering...")
    df = df.drop_duplicates(subset=['ID_Cliente']).reset_index(drop=True)
    df['Antiguedad_Meses'] = df['Antiguedad_Meses'].astype(str).str.extract(r'(\d+)').astype(float)
    
    mediana_tickets = df['Soporte_Tickets'].median()
    df['Soporte_Tickets'] = df['Soporte_Tickets'].fillna(mediana_tickets)

    mediana_edad = df[(df['Edad'] >= 18) & (df['Edad'] <= 90)]['Edad'].median()
    df['Edad'] = np.where((df['Edad'] < 18) | (df['Edad'] > 90), mediana_edad, df['Edad'])

    # Creación de variables
    df['Tickets_Por_Mes'] = df['Soporte_Tickets'] / (df['Antiguedad_Meses'] + 1)
    df['Uso_Categoria'] = pd.qcut(df['Uso_Diario_Min'], q=3, labels=['Bajo', 'Medio', 'Alto'])
    df['Costo_Por_Minuto'] = df['Monto_Mensual'] / (df['Uso_Diario_Min'] + 1)

    df_procesado = pd.get_dummies(df, columns=['Genero', 'Region', 'Uso_Categoria'], drop_first=True)
    
    os.makedirs('../data', exist_ok=True)
    df_procesado.to_csv('../data/datacloud_churn_clean.csv', index=False)

    print("⚙️ [3/4] Entrenando modelo con SMOTE...")
    X = df_procesado.drop(columns=['ID_Cliente', 'Churn'])
    y = df_procesado['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    rf_balanced = RandomForestClassifier(n_estimators=150, class_weight='balanced', random_state=42)
    rf_balanced.fit(X_train_res, y_train_res)

    print("💾 [4/4] Guardando modelos y artefactos...")
    os.makedirs('../models', exist_ok=True)
    joblib.dump(rf_balanced, '../models/churn_model_rf.pkl')
    joblib.dump(X.columns.tolist(), '../models/model_features.pkl')

    print("✅ ¡Pipeline ejecutado con éxito! Artefactos actualizados en 'models/'.")

if __name__ == "__main__":
    ejecutar_pipeline()