# ------------------------------------------------------------
# Reto Titanic — Dashboard Interactivo (Streamlit)
# 4 secciones: Exploración, Predicción, Análisis de Modelos, What-If
# Ejecuta con: streamlit run dashboard_titanic_clean.py
# Requisitos: pip install streamlit pandas numpy scikit-learn shap plotly joblib dill
# ------------------------------------------------------------

import os
import json
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# Opcionales
try:
    import shap
except Exception:
    shap = None

try:
    import plotly.express as px
except Exception:
    px = None

st.set_page_config(page_title="Reto Titanic — Dashboard Interactivo", layout="wide")

# ==============================
# Utilidades con cache
# ==============================
@st.cache_data(show_spinner=False)
def load_data(csv_path: str):
    if not csv_path or not os.path.exists(csv_path):
        return None
    try:
        return pd.read_csv(csv_path)
    except Exception as e:
        st.error(f"Error al cargar {csv_path}: {e}")
        return None

@st.cache_resource(show_spinner=False)
def load_model(pkl_path: str):
    """Carga tolerante de modelos serializados.
    Intenta en orden: pickle -> joblib -> dill.
    """
    if not pkl_path or not os.path.exists(pkl_path):
        return None
    # 1) intentar con pickle
    try:
        with open(pkl_path, "rb") as f:
            return pickle.load(f)
    except Exception as e_pickle:
        # 2) intentar con joblib
        try:
            import joblib
            return joblib.load(pkl_path)
        except Exception as e_joblib:
            # 3) intentar con dill
            try:
                import dill
                with open(pkl_path, "rb") as f:
                    return dill.load(f)
            except Exception as e_dill:
                st.error(
                    f"Error al cargar {pkl_path}.\n"
                    f"- pickle: {e_pickle}\n"
                    f"- joblib: {e_joblib}\n"
                    f"- dill: {e_dill}\n"
                    "Sugerencia: guarda el modelo como Pipeline y usa el mismo método (pickle/joblib) para cargarlo."
                )
                return None

# ----------------------------
# Adaptador: genera features que el modelo espera
# ----------------------------
def _titanic_age_group(age: float) -> str:
    if age is None or pd.isna(age):
        return "Adulto Joven"  # fallback
    if age < 12:
        return "Niño"
    if age < 18:
        return "Adolescente"
    if age < 40:
        return "Adulto Joven"
    return "Mayor"

def preprocess_for_model(model, X_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Construye un DataFrame con las columnas EXACTAS que el modelo espera
    (usando model.feature_names_in_ si existe). Para columnas no presentes
    en X_raw se generan a partir de reglas típicas del Titanic.
    """
    expected = list(getattr(model, "feature_names_in_", []))
    if not expected:
        # El modelo no expone nombres -> probablemente es Pipeline; usar tal cual
        return X_raw

    row = X_raw.iloc[0]
    feats = {}

    # Passthrough de numéricas si el modelo las espera
    for col in ["Pclass", "Age", "SibSp", "Parch", "Fare"]:
        if col in expected and col in X_raw.columns:
            feats[col] = float(row[col])

    # Sex crudo u one-hot
    if "Sex" in expected and "Sex" in X_raw:
        feats["Sex"] = row["Sex"]
    for s in ["male", "female"]:
        name = f"Sex_{s}"
        if name in expected and "Sex" in X_raw:
            feats[name] = 1 if row["Sex"] == s else 0

    # Embarked crudo u one-hot
    if "Embarked" in expected and "Embarked" in X_raw:
        feats["Embarked"] = row["Embarked"]
    for e in ["S", "C", "Q"]:
        name = f"Embarked_{e}"
        if name in expected and "Embarked" in X_raw:
            feats[name] = 1 if row["Embarked"] == e else 0

    # Grupos de edad
    grp = _titanic_age_group(float(row["Age"]) if "Age" in X_raw else np.nan)
    for age_group in ["Niño", "Adolescente", "Adulto Joven", "Mayor"]:
        name = f"AgeGroup_{age_group}"
        if name in expected:
            feats[name] = 1 if age_group == grp else 0

    # Familia
    famsize = (int(row["SibSp"]) if "SibSp" in X_raw else 0) + (int(row["Parch"]) if "Parch" in X_raw else 0) + 1
    if "FamilySize" in expected:
        feats["FamilySize"] = famsize
    if "IsAlone" in expected:
        feats["IsAlone"] = 1 if famsize == 1 else 0

    # CabinKnown
    if "CabinKnown" in expected:
        feats["CabinKnown"] = 0

    # Construir DataFrame final con listas
    data_dict = {}
    for col in expected:
        if col in feats:
            data_dict[col] = [feats[col]]
        else:
            data_dict[col] = [0]
    X_out = pd.DataFrame(data_dict)

    # Asegurar orden correcto
    X_out = X_out[expected]

    return X_out

# ==============================
# Rutas por defecto 
# ==============================
DEFAULT_DATA = "data_p1.csv"
DEFAULT_MODELS = {
    "Logistic Regression": "LogisticRegression_final.pkl",
    "Random Forest": "RandomForest_final.pkl",
    "XGBoost/LightGBM": "XGBoost_final.pkl",
    "SVM": "SVM_final.pkl",
}

# ==============================
# Sidebar: configuración
# ==============================
st.title("Reto Titanic — Dashboard Interactivo")

with st.sidebar:
    st.header("Configuración")
    data_path = st.text_input("Ruta del CSV limpio", value=DEFAULT_DATA)
    df = load_data(data_path)

    st.markdown("---")
    st.subheader("Modelos")
    models_loaded = {}
    for name, default_path in DEFAULT_MODELS.items():
        path = st.text_input(name, value=default_path, key=f"model_{name}")
        model = load_model(path)
        status = "OK" if model is not None else "X"
        st.caption(f"{status} {path}")
        models_loaded[name] = model

    st.markdown("---")
    page = st.radio("Navegar a:", ["Exploración", "Predicción", "Análisis de Modelos", "What-If"])

if df is None:
    st.info("Carga un CSV (ej. data_p1.csv) para habilitar las secciones basadas en datos.")

# ==============================
# Página: Exploración
# ==============================
def page_exploracion():
    st.header("Exploración (EDA)")
    if df is None:
        st.info("Carga un CSV para habilitar el EDA.")
        return

    with st.expander("Filtros", expanded=True):
        # Filtros directos sin selectores de columnas
        
        # 1. Sexo - filtro directo
        try:
            if "Sex" in df.columns:
                sex_opts = df["Sex"].dropna().unique().tolist()
                sex_opts = sorted([str(x) for x in sex_opts])
                selected_sex = st.multiselect("Sexo", options=sex_opts, default=[])
            else:
                selected_sex = []
                st.warning("Columna 'Sex' no encontrada")
        except Exception:
            selected_sex = []
            st.warning("Error al procesar columna Sex")

        # 2. Clase - filtro directo  
        try:
            if "Pclass" in df.columns:
                pclass_opts = df["Pclass"].dropna().unique().tolist()
                pclass_opts = sorted([str(x) for x in pclass_opts])
                selected_pclass = st.multiselect("Clase", options=pclass_opts, default=[])
            else:
                selected_pclass = []
                st.warning("Columna 'Pclass' no encontrada")
        except Exception:
            selected_pclass = []
            st.warning("Error al procesar columna Pclass")

        # 3. Solo rango de edad (sin selector de columna)
        try:
            if "Age" in df.columns and pd.api.types.is_numeric_dtype(df["Age"]):
                valid_ages = df["Age"].dropna()
                if len(valid_ages) > 0:
                    age_min, age_max = float(valid_ages.min()), float(valid_ages.max())
                else:
                    age_min, age_max = 0.0, 80.0
                selected_age = st.slider("Rango de edad", 
                                        min_value=age_min, 
                                        max_value=age_max, 
                                        value=(age_min, age_max))
            else:
                selected_age = (0.0, 80.0)
                st.warning("Columna 'Age' no encontrada o no es numérica")
        except Exception:
            selected_age = (0.0, 80.0)
            st.warning("Error al procesar rango de edad")

        # 4. Familia simplificada (configuración fija)
        if st.checkbox("Filtrar por tamaño de familia"):
            family_size_filter = st.selectbox(
                "Tamaño de familia", 
                options=["Todos", "Solo (sin familia)", "Con familia pequeña (2-4)", "Con familia grande (5+)"]
            )
        else:
            family_size_filter = "Todos"

    # Aplicar filtros
    dff = df.copy()
    try:
        # Filtro por sexo
        if selected_sex and len(selected_sex) > 0 and "Sex" in dff.columns:
            dff = dff[dff["Sex"].astype(str).isin(selected_sex)]
        
        # Filtro por clase
        if selected_pclass and len(selected_pclass) > 0 and "Pclass" in dff.columns:
            dff = dff[dff["Pclass"].astype(str).isin(selected_pclass)]
        
        # Filtro por edad
        if "Age" in dff.columns and pd.api.types.is_numeric_dtype(dff["Age"]):
            dff = dff[dff["Age"].between(selected_age[0], selected_age[1])]
        
        # Filtro por familia simplificado
        if family_size_filter != "Todos":
            if "SibSp" in dff.columns and "Parch" in dff.columns:
                dff["FamilySize"] = dff["SibSp"] + dff["Parch"] + 1
                
                if family_size_filter == "Solo (sin familia)":
                    dff = dff[dff["FamilySize"] == 1]
                elif family_size_filter == "Con familia pequeña (2-4)":
                    dff = dff[dff["FamilySize"].between(2, 4)]
                elif family_size_filter == "Con familia grande (5+)":
                    dff = dff[dff["FamilySize"] >= 5]
            else:
                st.warning("Columnas SibSp/Parch no encontradas para filtro de familia")

    except Exception as e:
        st.warning(f"Error aplicando filtros: {e}")
        dff = df.copy()

    # Mostrar tabla filtrada
    st.subheader("Tabla filtrada")
    
    # Mostrar estadísticas de supervivencia
    if len(dff) > 0 and "Survived" in dff.columns:
        survived_count = dff["Survived"].sum()
        total_count = len(dff)
        survival_rate = (survived_count / total_count * 100) if total_count > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total registros", f"{total_count:,}")
        with col2:
            st.metric("Sobrevivieron", f"{int(survived_count):,}")
        with col3:
            st.metric("No sobrevivieron", f"{int(total_count - survived_count):,}")
        with col4:
            st.metric("Tasa supervivencia", f"{survival_rate:.1f}%")
    else:
        st.write(f"Mostrando {len(dff):,} de {len(df):,} registros")
    
    if len(dff) > 0:
        st.dataframe(dff)
    else:
        st.warning("No hay datos que mostrar con los filtros aplicados.")
        return

    # Visualizaciones
    if px is not None:
        st.subheader("Distribuciones")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Gráfico de sexo
            if "Sex" in dff.columns:
                try:
                    fig1 = px.histogram(dff, x="Sex", title="Distribución por Sexo")
                    st.plotly_chart(fig1, use_container_width=True)
                except Exception as e:
                    st.warning(f"Error en gráfico de sexo: {e}")
        
        with col2:
            # Gráfico de clase
            if "Pclass" in dff.columns:
                try:
                    fig2 = px.histogram(dff, x="Pclass", title="Distribución por Clase")
                    st.plotly_chart(fig2, use_container_width=True)
                except Exception as e:
                    st.warning(f"Error en gráfico de clase: {e}")

        # Análisis de supervivencia
        if "Survived" in dff.columns:
            st.subheader("Análisis de supervivencia")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Supervivencia por sexo
                if "Sex" in dff.columns:
                    try:
                        surv_sex = dff.groupby("Sex")["Survived"].agg(['mean', 'count']).reset_index()
                        surv_sex['mean'] = surv_sex['mean'] * 100  # Convertir a porcentaje
                        fig3 = px.bar(surv_sex, x="Sex", y="mean", 
                                     title="Tasa de supervivencia por Sexo (%)",
                                     text="count")
                        fig3.update_traces(texttemplate='n=%{text}', textposition="outside")
                        fig3.update_layout(yaxis_title="Supervivencia (%)")
                        st.plotly_chart(fig3, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Error en supervivencia por sexo: {e}")
            
            with col2:
                # Supervivencia por clase
                if "Pclass" in dff.columns:
                    try:
                        surv_class = dff.groupby("Pclass")["Survived"].agg(['mean', 'count']).reset_index()
                        surv_class['mean'] = surv_class['mean'] * 100
                        fig4 = px.bar(surv_class, x="Pclass", y="mean", 
                                     title="Tasa de supervivencia por Clase (%)",
                                     text="count")
                        fig4.update_traces(texttemplate='n=%{text}', textposition="outside")
                        fig4.update_layout(yaxis_title="Supervivencia (%)")
                        st.plotly_chart(fig4, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Error en supervivencia por clase: {e}")

            # Heatmap de supervivencia por sexo y clase
            if "Sex" in dff.columns and "Pclass" in dff.columns:
                try:
                    surv_heatmap = dff.groupby(["Sex", "Pclass"])["Survived"].mean().unstack()
                    surv_heatmap = surv_heatmap * 100  # Convertir a porcentaje
                    
                    fig5 = px.imshow(
                        surv_heatmap, 
                        title="Tasa de supervivencia: Sexo vs Clase (%)",
                        aspect="auto",
                        color_continuous_scale="RdYlGn"
                    )
                    fig5.update_xaxes(title="Clase")
                    fig5.update_yaxes(title="Sexo")
                    st.plotly_chart(fig5, use_container_width=True)
                except Exception as e:
                    st.warning(f"Error en heatmap: {e}")

    else:
        st.info("Instala plotly para ver visualizaciones: pip install plotly")

# ==============================
# Página: Predicción (CORREGIDA)
# ==============================
def page_prediccion():
    st.header("Predicción con explicación")

    disponibles = [k for k, v in models_loaded.items() if v is not None]
    if not disponibles:
        st.info("No hay modelos cargados. Revisa las rutas en el panel izquierdo.")
        return

    model_name = st.selectbox("Modelo", options=disponibles)
    model = models_loaded[model_name]

    st.markdown("Ingreso de características (usa nombres crudos del Titanic)")
    c1, c2, c3 = st.columns(3)
    with c1:
        pclass = st.selectbox("Pclass", [1, 2, 3], index=2)
        sex = st.selectbox("Sex", ["male", "female"])
    with c2:
        age = st.number_input("Age", min_value=0.0, max_value=100.0, value=30.0, step=1.0)
        sibsp = st.number_input("SibSp", min_value=0, max_value=10, value=0, step=1)
    with c3:
        parch = st.number_input("Parch", min_value=0, max_value=10, value=0, step=1)
        fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=32.2, step=0.1)
    embarked = st.selectbox("Embarked", ["S", "C", "Q"])

    X_input = pd.DataFrame({
        "Pclass": [pclass],
        "Sex": [sex],
        "Age": [age],
        "SibSp": [sibsp],
        "Parch": [parch],
        "Fare": [fare],
        "Embarked": [embarked],
    })

    if st.button("Predecir"):
        try:
            # ADAPTADOR
            X_model = preprocess_for_model(model, X_input)

            proba = model.predict_proba(X_model)[0, 1] if hasattr(model, "predict_proba") else None
            pred = model.predict(X_model)[0]

            c1, c2 = st.columns(2)
            with c1:
                st.metric("Probabilidad de supervivencia", f"{(proba*100):.2f}%" if proba is not None else "N/A")
                st.metric("Predicción", "Sobrevive (1)" if int(pred) == 1 else "No sobrevive (0)")
            with c2:
                st.json(X_input.to_dict(orient="records")[0])

            if shap is not None:
                try:
                    st.markdown("Explicación SHAP (local)")
                    
                    # Intenta crear un explainer directo primero
                    try:
                        explainer = shap.Explainer(model)
                        vals = explainer(X_model)
                    except Exception:
                        # Si falla, usa KernelExplainer con configuración correcta
                        st.info("Usando KernelExplainer (puede tardar unos segundos)...")
                        
                        # Crear datos de background más robustos
                        if df is not None and len(df) > 10:
                            # Usa una muestra del dataset original como background
                            background_sample = df.sample(min(50, len(df)), random_state=42)
                            background_processed = preprocess_for_model(model, background_sample)
                        else:
                            # Crea background sintético si no hay datos
                            background_processed = X_model.copy()
                            
                        # Define funciones wrapper correctas
                        if hasattr(model, "predict_proba"):
                            def model_predict(X):
                                # Asegura que devuelve solo las probabilidades de clase positiva
                                return model.predict_proba(X)[:, 1]
                        else:
                            def model_predict(X):
                                # Para modelos sin predict_proba
                                return model.predict(X).astype(float)
                        
                        # Crear explainer con background adecuado
                        explainer = shap.KernelExplainer(
                            model_predict, 
                            background_processed,
                            link="identity"
                        )
                        
                        # Generar explicaciones
                        vals = explainer.shap_values(X_model, nsamples=50)
                        
                        # Convertir a formato estándar si es necesario
                        if not hasattr(vals, 'values'):
                            # vals es un array numpy, crear objeto similar a shap.Explanation
                            vals = type('obj', (object,), {
                                'values': vals.reshape(1, -1),
                                'base_values': explainer.expected_value,
                                'data': X_model.values
                            })()
                    
                    # Mostrar explicaciones
                    st.write("Valores SHAP (tabla)")
                    if hasattr(vals, 'values'):
                        shap_df = pd.DataFrame(
                            vals.values, 
                            columns=X_model.columns,
                            index=['Contribución SHAP']
                        )
                    else:
                        shap_df = pd.DataFrame(
                            vals.reshape(1, -1), 
                            columns=X_model.columns,
                            index=['Contribución SHAP']
                        )
                    
                    st.dataframe(shap_df)
                    
                    # Interpretación simple
                    st.markdown("**Interpretación:**")
                    shap_vals = shap_df.iloc[0]
                    positive_features = shap_vals[shap_vals > 0].sort_values(ascending=False)
                    negative_features = shap_vals[shap_vals < 0].sort_values(ascending=True)
                    
                    if len(positive_features) > 0:
                        st.write("**Factores que aumentan supervivencia:**")
                        for feat, val in positive_features.head(3).items():
                            st.write(f"- {feat}: +{val:.4f}")
                    
                    if len(negative_features) > 0:
                        st.write("**Factores que disminuyen supervivencia:**")
                        for feat, val in negative_features.head(3).items():
                            st.write(f"- {feat}: {val:.4f}")
                            
                except Exception as e:
                    st.warning(f"No se pudo generar SHAP: {e}")
                    st.info("Esto puede ocurrir si el modelo no es compatible con SHAP o hay problemas dimensionales.")

            st.success("Predicción generada.")
        except Exception as e:
            st.error(f"Error al predecir: {e}")
            st.info("Si guardaste solo el clasificador y NO el pipeline, debes aplicar el mismo preprocesamiento antes de predecir.")

# ==============================
# Página: Análisis de Modelos
# ==============================
def page_analisis_modelos():
    st.header("Análisis y comparación de modelos")
    st.write("Sube una tabla de métricas (CSV/JSON) para compararlas de forma interactiva.")

    c1, c2 = st.columns(2)
    with c1:
        uploaded = st.file_uploader("Tabla comparativa (CSV/JSON)", type=["csv", "json"])
    with c2:
        st.caption("Formato sugerido:")
        st.code("Modelo,Accuracy,Precision,Recall,F1,ROC_AUC,Tiempo_Train,Tiempo_Infer")

    if uploaded is not None:
        try:
            table = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.DataFrame(json.load(uploaded))
            st.subheader("Tabla de métricas")
            st.dataframe(table)
            if px is not None:
                st.subheader("Gráfico comparativo")
                metric = st.selectbox("Métrica", options=[c for c in table.columns if c != "Modelo"])
                st.plotly_chart(px.bar(table, x="Modelo", y=metric, title=f"Comparación por {metric}"), use_container_width=True)
        except Exception as e:
            st.error(f"Error al procesar métricas: {e}")

    st.markdown("---")
    st.subheader("Importancias de características (opcional)")
    up_imp = st.file_uploader("Importancias (CSV/JSON)", type=["csv", "json"], key="imp")
    if up_imp is not None:
        try:
            imp = pd.read_csv(up_imp) if up_imp.name.endswith(".csv") else pd.DataFrame(json.load(up_imp))
            st.dataframe(imp)
            if px is not None and {"feature", "importance"}.issubset(set(imp.columns)):
                fig = px.bar(
                    imp.sort_values("importance", ascending=False).head(20),
                    x="importance", y="feature", orientation="h",
                    title="Top 20 features"
                )
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error al procesar importancias: {e}")

# ==============================
# Página: What-If
# ==============================
def page_what_if():
    st.header("What-If (análisis contrafactual)")

    disponibles = [k for k, v in models_loaded.items() if v is not None]
    if not disponibles:
        st.info("No hay modelos cargados.")
        return

    model_name = st.selectbox("Modelo", options=disponibles, key="whatif_model")
    model = models_loaded[model_name]

    st.markdown("Define un pasajero base y modifica características para ver cambios en la probabilidad.")

    c1, c2, c3 = st.columns(3)
    with c1:
        pclass = st.selectbox("Pclass", [1, 2, 3], index=2, key="w_pclass")
        sex = st.selectbox("Sex", ["male", "female"], key="w_sex")
    with c2:
        age = st.number_input("Age", 0.0, 100.0, 30.0, 1.0, key="w_age")
        sibsp = st.number_input("SibSp", 0, 10, 0, 1, key="w_sibsp")
    with c3:
        parch = st.number_input("Parch", 0, 10, 0, 1, key="w_parch")
        fare = st.number_input("Fare", 0.0, 600.0, 32.2, 0.1, key="w_fare")
    
    embarked = st.selectbox("Embarked", ["S", "C", "Q"], key="w_embarked")

    X_base = pd.DataFrame({
        "Pclass": [pclass],
        "Sex": [sex],
        "Age": [age],
        "SibSp": [sibsp],
        "Parch": [parch],
        "Fare": [fare],
        "Embarked": [embarked],
    })

    try:
        X_model = preprocess_for_model(model, X_base)
        proba_base = model.predict_proba(X_model)[0, 1] if hasattr(model, "predict_proba") else None
        
        if proba_base is not None:
            st.metric("Probabilidad base de supervivencia", f"{(proba_base*100):.2f}%")
        
        st.subheader("Análisis de sensibilidad")
        st.markdown("Modifica una característica y ve cómo cambia la predicción:")
        
        # Análisis por edad
        if st.checkbox("Analizar impacto de la edad"):
            ages = np.arange(0, 81, 5)
            probas = []
            
            for test_age in ages:
                X_test = X_base.copy()
                X_test["Age"] = [test_age]
                X_test_model = preprocess_for_model(model, X_test)
                if hasattr(model, "predict_proba"):
                    prob = model.predict_proba(X_test_model)[0, 1]
                    probas.append(prob * 100)
                else:
                    probas.append(model.predict(X_test_model)[0] * 100)
            
            if px is not None:
                fig = px.line(x=ages, y=probas, title="Probabilidad de supervivencia vs Edad")
                fig.update_xaxes(title="Edad")
                fig.update_yaxes(title="Probabilidad (%)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Instala plotly para visualizar la sensibilidad: pip install plotly")
        
        # Análisis por tarifa
        if st.checkbox("Analizar impacto de la tarifa"):
            fares = np.arange(0, 201, 10)
            probas_fare = []
            
            for test_fare in fares:
                X_test = X_base.copy()
                X_test["Fare"] = [float(test_fare)]
                X_test_model = preprocess_for_model(model, X_test)
                if hasattr(model, "predict_proba"):
                    prob = model.predict_proba(X_test_model)[0, 1]
                    probas_fare.append(prob * 100)
                else:
                    probas_fare.append(model.predict(X_test_model)[0] * 100)
            
            if px is not None:
                fig = px.line(x=fares, y=probas_fare, title="Probabilidad de supervivencia vs Tarifa")
                fig.update_xaxes(title="Tarifa")
                fig.update_yaxes(title="Probabilidad (%)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Instala plotly para visualizar la sensibilidad: pip install plotly")
                
    except Exception as e:
        st.error(f"Error en análisis What-If: {e}")

# ==============================
# Router principal
# ==============================
if page == "Exploración":
    page_exploracion()
elif page == "Predicción":
    page_prediccion()
elif page == "Análisis de Modelos":
    page_analisis_modelos()
elif page == "What-If":
    page_what_if()
