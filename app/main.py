import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Sistema Predictivo & Prescriptivo de Churn", layout="wide")

st.title("📊 DataCloud Solutions - Sistema Integral de Churn & Segmentación")
st.markdown("Plataforma interactiva para predecir abandono, asignar clusters de comportamiento y analizar explicabilidad (XAI).")

@st.cache_resource
def load_all_assets():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    rf_model = joblib.load(os.path.join(BASE_DIR, 'models', 'churn_model_rf.pkl'))
    rf_features = joblib.load(os.path.join(BASE_DIR, 'models', 'model_features.pkl'))
    
    kmeans_model = joblib.load(os.path.join(BASE_DIR, 'models', 'kmeans_model.pkl'))
    cluster_scaler = joblib.load(os.path.join(BASE_DIR, 'models', 'cluster_scaler.pkl'))
    cluster_features = joblib.load(os.path.join(BASE_DIR, 'models', 'cluster_features.pkl'))
    
    return rf_model, rf_features, kmeans_model, cluster_scaler, cluster_features

try:
    rf_model, rf_features, kmeans_model, cluster_scaler, cluster_features = load_all_assets()

    tab1, tab2, tab3 = st.tabs(["🔮 Evaluador de Cliente", "🧩 Perfiles & Clustering", "⚖️ Explicabilidad & Ética"])

    with tab1:
        st.subheader("Evaluación Individual de Riesgo")
        col1, col2 = st.columns(2)
        
        with col1:
            edad = st.number_input("Edad del Cliente", min_value=18, max_value=90, value=35)
            antiguedad = st.number_input("Antigüedad (Meses)", min_value=1, max_value=120, value=24)
            soporte_tickets = st.number_input("Tickets de Soporte", min_value=0, max_value=20, value=3)
            uso_minutos = st.number_input("Uso Diario (Minutos)", min_value=0.0, max_value=600.0, value=45.0)
        
        with col2:
            monto_mensual = st.number_input("Monto Mensual ($USD)", min_value=10.0, max_value=1000.0, value=150.0)
            genero = st.selectbox("Género", ["M", "F", "Otro"])
            region = st.selectbox("Región", ["Norteamérica", "LATAM", "Europa", "APAC"])

        if st.button("Evaluar Cliente", type="primary"):
            tickets_por_mes = soporte_tickets / (antiguedad + 1)
            costo_por_minuto = monto_mensual / (uso_minutos + 1)
            
            input_dict = {
                'Edad': edad,
                'Antiguedad_Meses': antiguedad,
                'Soporte_Tickets': soporte_tickets,
                'Uso_Diario_Min': uso_minutos,
                'Monto_Mensual': monto_mensual,
                'Tickets_Por_Mes': tickets_por_mes,
                'Costo_Por_Minuto': costo_por_minuto,
                'Genero_M': 1 if genero == 'M' else 0,
                'Genero_Otro': 1 if genero == 'Otro' else 0,
                'Region_Europa': 1 if region == 'Europa' else 0,
                'Region_LATAM': 1 if region == 'LATAM' else 0,
                'Region_Norteamérica': 1 if region == 'Norteamérica' else 0,
                'Uso_Categoria_Medio': 1 if 30 <= uso_minutos <= 90 else 0,
                'Uso_Categoria_Alto': 1 if uso_minutos > 90 else 0,
            }
            
            df_input = pd.DataFrame([input_dict]).reindex(columns=rf_features, fill_value=0)
            prob_churn = rf_model.predict_proba(df_input)[0][1]
            pred_churn = int(prob_churn >= 0.35)
            
            # Predicción del Cluster No Supervisado
            df_cluster_input = pd.DataFrame([[edad, antiguedad, soporte_tickets, uso_minutos, monto_mensual, costo_por_minuto]], columns=cluster_features)
            scaled_cluster_input = cluster_scaler.transform(df_cluster_input)
            cluster_id = kmeans_model.predict(scaled_cluster_input)[0]
            
            nombres_clusters = {
                0: "Cluster 0: Clientes de Bajo Uso y Alto Soporte (Alto Riesgo)",
                1: "Cluster 1: Usuarios Corporativos Intensivos (Alta Retención)",
                2: "Cluster 2: Clientes Estándar / Uso Moderado"
            }

            st.divider()
            c1, c2 = st.columns(2)
            c1.metric("Probabilidad de Churn", f"{prob_churn * 100:.1f}%")
            c2.info(f"**Segmento Asignado:** {nombres_clusters.get(cluster_id, 'Cluster ' + str(cluster_id))}")
            
            if pred_churn == 1:
                st.error("⚠️ ALTO RIESGO DE CANCELACIÓN: Activar protocolo de fidelización y revisión de tickets.")
            else:
                st.success("✅ CLIENTE ESTABLE: Continuar monitoreo regular.")

    with tab2:
        st.subheader("Segmentación K-Means de Clientes")
        st.markdown("""
        * **Cluster 0:** Usuarios con alta fricción (muchos tickets de soporte y bajo uso).
        * **Cluster 1:** Usuarios corporativos maduros y de alto valor.
        * **Cluster 2:** Clientes estándar con comportamiento equilibrado.
        """)

    with tab3:
        st.subheader("Explicabilidad (XAI) & Protocolo Ético")
        st.markdown("""
        * **SHAP Summary:** Identifica que 'Soporte_Tickets', 'Uso_Diario_Min' y 'Costo_Por_Minuto' son los principales impulsores globales del abandono.
        * **Evaluación de Sesgo (Fairness):** Las variables demográficas (`Genero` y `Region`) no ejercen un peso discriminatorio en las decisiones del modelo supervisado.
        """)

except Exception as e:
    st.error(f"Error al cargar modelos: {e}")