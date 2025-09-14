# Modelado Predictivo de Supervivencia en el Titanic

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge">
  <img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/ivanlps/titanic-ml-project?style=for-the-badge&logo=github">
</p>

## 📄 Descripción del Proyecto

Este proyecto presenta un análisis de datos y modelado predictivo utilizando el clásico conjunto de datos de los pasajeros del Titanic. El objetivo principal es construir un modelo capaz de predecir la probabilidad de supervivencia de un pasajero basándose en sus características demográficas y de viaje.

El análisis no solo se enfoca en la creación de un modelo preciso, sino también en la **interpretabilidad** de sus decisiones y un profundo **análisis de fairness** para detectar posibles sesgos en las predicciones.

---

## 📊 Fases del Análisis

* **Análisis Exploratorio de Datos (EDA):** Identificación de patrones y relaciones clave en los datos, como la influencia del género, la clase y la edad en la supervivencia.
* **Preprocesamiento e Ingeniería de Características:** Limpieza de datos, imputación de valores faltantes y creación de nuevas variables para mejorar el rendimiento del modelo.
* **Modelado y Evaluación:** Entrenamiento y comparación de cuatro modelos de clasificación (Regresión Logística, Random Forest, SVM y XGBoost) para seleccionar el de mejor desempeño.
* **Interpretabilidad del Modelo:** Uso de SHAP para entender qué características influyen más en las predicciones y cómo lo hacen, tanto a nivel global como individual.
* **Análisis de Equidad (Fairness):** Evaluación de sesgos demográficos en las predicciones del modelo, analizando métricas como la disparidad demográfica y la igualdad de oportunidad para distintos grupos.

---

## ⚙️ Tecnologías Utilizadas

El proyecto fue desarrollado enteramente en Python y se apoya en las siguientes librerías:

* **Análisis de Datos:** Pandas, NumPy
* **Visualización:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-learn, XGBoost
* **Interpretabilidad:** SHAP
* **Gestión de Modelos:** Joblib

---

## 📁 Estructura del Repositorio

El proyecto está organizado siguiendo una estructura clara para facilitar su comprensión y reproducibilidad.

```
.
├── data/              # Contiene los datasets (crudo y procesado)
├── notebooks/         # Jupyter Notebooks para cada fase del análisis (EDA, Modelado, etc.)
├── paper/             # Reporte generado (PDF)
├── results/           # Visualizaciones y tablas que contienen los hallazgos y resultados más pertinentes 
├── src/               # Scripts de Python (.py) con código modularizado
├── models/            # Modelos entrenados y guardados (.pkl)
├── LICENSE            # Licencia del proyecto (MIT)
└── README.md          # Este archivo
```

---

## ✨ Resultados Clave

* El modelo con mejor desempeño fue **Random Forest**, alcanzando una **precisión (accuracy) del 81%** y un **F1-Score de 0.77**.
* Las características más influyentes para predecir la supervivencia fueron, en orden de importancia: **género**, **tarifa pagada** por persona y **clase del pasajero**.
* El análisis de fairness reveló que el modelo reproduce los sesgos históricos presentes en los datos, mostrando por ejemplo una probabilidad significativamente mayor de predecir la supervivencia para las mujeres que para los hombres.

---

## ⚖️ Licencia

Este proyecto se distribuye bajo la **Licencia MIT**. Consulta el archivo `LICENSE` para más detalles.
