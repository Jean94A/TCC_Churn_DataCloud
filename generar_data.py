import numpy as np
import pandas as pd
import os

# Crear la carpeta data si no existe
os.makedirs('data', exist_ok=True)

np.random.seed(42)
n = 10000

# 1. ID_Cliente (con duplicados y nulos)
ids = [f"CLI-{1000+i}" for i in range(n-50)]
ids.extend(ids[:40])
ids.extend([np.nan]*10)
np.random.shuffle(ids)

# 2. Edad (con outliers: negativos y 150 años)
edad = np.random.normal(38, 10, n).astype(int)
edad[np.random.choice(n, 30, replace=False)] = np.random.choice([-5, -1, 145, 150], 30)

# 3. Antiguedad Meses (texto y números mezclados)
antiguedad = np.random.randint(1, 72, n).astype(object)
idx_texto = np.random.choice(n, 200, replace=False)
antiguedad[idx_texto] = [f"{x} meses" for x in antiguedad[idx_texto]]

# 4. Soporte_Tickets (faltantes NAs)
soporte = np.random.poisson(2, n).astype(float)
idx_na = np.random.choice(n, 500, replace=False)
soporte[idx_na] = np.nan

# 5. Uso Diario Min (sesgado)
uso_min = np.random.exponential(45, n)

# 6. Monto Mensual (tarifas $USD)
monto = np.random.uniform(29.99, 299.99, n)

# 7. Genero / Region (categóricas)
genero = np.random.choice(['M', 'F', 'Otro'], n, p=[0.48, 0.48, 0.04])
region = np.random.choice(['Norteamérica', 'LATAM', 'Europa', 'APAC'], n, p=[0.50, 0.25, 0.15, 0.10])

# 8. Churn (80% activo 0, 20% canceló 1)
prob_churn = 1 / (1 + np.exp(-(-2.0 + 0.3*np.nan_to_num(soporte) - 0.02*uso_min + 0.005*monto)))
churn = (np.random.rand(n) < prob_churn).astype(int)

df_churn = pd.DataFrame({
    'ID_Cliente': ids,
    'Edad': edad,
    'Antiguedad_Meses': antiguedad,
    'Soporte_Tickets': soporte,
    'Uso_Diario_Min': uso_min,
    'Monto_Mensual': monto,
    'Genero': genero,
    'Region': region,
    'Churn': churn
})

df_churn.to_csv('data/datacloud_churn.csv', index=False)
print("¡Archivo data/datacloud_churn.csv generado exitosamente!")