# 📊 Sistema Predictivo y Prescriptivo de Abandono de Clientes (Churn) — DataCloud Solutions

**Asignatura:** Ciencia de Datos II  
**Estudiante:** Jean Carlos Arrieta Arrieta  
**Metodología:** CRISP-DM  
**Caso Práctico:** DataCloud Solutions (Empresa B2B SaaS)

---

## 📌 Descripción del Proyecto

Este proyecto implementa una solución end-to-end de Ciencia de Datos para predecir, comprender y mitigar la tasa de cancelación voluntaria de clientes (*churn*) en la empresa **DataCloud Solutions**. 

Mediante el uso del marco **CRISP-DM**, la solución abarca desde el Análisis Exploratorio de Datos (EDA) y la imputación avanzada de calidad, hasta la construcción de modelos supervisados (*Random Forest* balanceado con SMOTE), segmentación no supervisada (*K-Means*) y explicabilidad del modelo con valores **SHAP**, todo desplegado en una aplicación interactiva construida en **Streamlit**.

---

## 📁 Estructura del Repositorio

```text
TCC_Churn_DataCloud/
├── app/
│   └── main.py                     # Aplicación web interactiva en Streamlit
├── data/
│   ├── datacloud_churn.csv         # Dataset original (10,000 registros)
│   └── datacloud_churn_clean.csv   # Dataset preprocesado y limpio
├── models/
│   ├── churn_model_rf.pkl          # Modelo Random Forest entrenado
│   ├── kmeans_model.pkl            # Modelo de Clustering K-Means
│   ├── cluster_scaler.pkl          # Escalador StandardScaler para K-Means
│   ├── cluster_features.pkl        # Variables seleccionadas para K-Means
│   └── model_features.pkl          # Variables seleccionadas para Random Forest
├── notebooks/
│   ├── 01_eda_fundamentos.ipynb    # EDA y diagnóstico de desbalance
│   ├── 02_preprocesamiento_features.ipynb # Limpieza, imputación y feature engineering
│   ├── 03_modelado_evaluacion.ipynb       # SMOTE, Random Forest y métricas
│   ├── 04_clustering_perfiles.ipynb       # Segmentación K-Means y perfiles
│   └── 05_explicabilidad_etica.ipynb      # SHAP Values y auditoría de equidad
├── src/
│   └── pipeline.py                 # Pipeline modular reproducible en Python
├── .gitignore                      # Exclusiones de control de versiones
├── generar_data.py                 # Script generador de datos sintéticos
└── requirements.txt                # Librerías y dependencias del proyecto