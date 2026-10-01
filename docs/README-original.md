# Proyecto Final – Machine Learning
## Predicción del Éxito Académico en Educación Superior
**Asignatura:** Aprendizaje Automático · Año académico 2025/2026  
**Institución:** Universidad Pontificia Comillas (ICAI)

---

## Estructura del proyecto

```
proyecto_ml/
├── main.py                        # Script principal (ejecuta todo)
├── README.md                      # Este archivo
├── data/
│   └── rendimiento_estudiantes.csv    # Dataset original
├── notebooks/
│   ├── 01_EDA_preprocesado.py     # Exploración y análisis descriptivo
│   ├── 02_clasificacion.py        # Tarea 1: Clasificación multiclase
│   ├── 03_regresion.py            # Tarea 2: Regresión
│   └── 04_no_supervisado.py       # Tarea 3: Aprendizaje no supervisado
└── outputs/                       # Figuras y CSVs generados automáticamente
```

---

## Requisitos

**Python:** 3.9 o superior  

### Instalación de dependencias

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn shap
```

O con el entorno recomendado:

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn shap
```

---

## Reproducción de resultados

### Opción 1: Ejecutar todo el proyecto de una vez

```bash
python main.py
```

Este comando ejecuta secuencialmente los cuatro módulos y genera todos los outputs.

### Opción 2: Ejecutar módulos individualmente

```bash
cd notebooks
python 01_EDA_preprocesado.py
python 02_clasificacion.py
python 03_regresion.py
python 04_no_supervisado.py
```

> **Importante:** El archivo `rendimiento_estudiantes.csv` debe estar en `data/`.

---

## Descripción de los módulos

### `01_EDA_preprocesado.py`
- Carga y visión general del dataset (4.424 estudiantes, 37 variables)
- Estadísticas descriptivas de variables numéricas y categóricas
- Análisis de valores nulos
- Distribución de la variable objetivo (Abandono / Matriculado / Graduado)
- Análisis univariante y bivariante por clase
- Matriz de correlación
- Preprocesado: codificación binaria + one-hot encoding

**Figuras generadas:**
- `fig01_distribucion_objetivo.png`
- `fig02_univariante_numericas.png`
- `fig03_correlacion.png`
- `fig04_notas_por_clase.png`
- `fig05_variables_socioeconomicas.png`

---

### `02_clasificacion.py`
**Objetivo:** Predecir el estado final del estudiante (multiclase)

**Modelos evaluados:**
| Modelo | Descripción |
|---|---|
| Regresión Logística | Modelo lineal de referencia (baseline) |
| Árbol de Decisión | Modelo interpretable no lineal |
| Random Forest | Ensemble de árboles (mejor rendimiento) |
| Gradient Boosting | Boosting secuencial |

**Métricas:** Accuracy, F1 Weighted, F1 Macro (justificación: dataset desbalanceado)

**Técnica para desbalanceo:** SMOTE en el conjunto de entrenamiento

**Figuras generadas:**
- `fig06_clasificacion_metricas.png` – Comparativa de modelos
- `fig07_confusion_matrix.png` – Matriz de confusión del mejor modelo
- `fig08_feature_importance_clf.png` – Importancia de variables (RF)

**Outputs CSV:**
- `clasificacion_metricas.csv`
- `clasificacion_predicciones.csv`

---

### `03_regresion.py`
**Objetivo:** Predecir `nota_media_2sem` (calificación media del 2º semestre)

**Variables excluidas por data leakage:**
| Variable | Motivo de exclusión |
|---|---|
| `asignaturas_2sem_matriculadas` | Contemporánea al target |
| `asignaturas_2sem_evaluadas` | Contemporánea al target |
| `asignaturas_2sem_aprobadas` | Directamente correlada con la nota |
| `asignaturas_2sem_sin_evaluacion` | Misma temporalidad |
| `asignaturas_2sem_convalidadas` | Misma temporalidad |
| `objetivo` | Usa información del resultado final |

**Modelos evaluados:** Regresión Lineal, Ridge, Lasso, Árbol, Random Forest, Gradient Boosting

**Métricas:** RMSE, MAE, R², CV R² (5-fold)

**Interpretabilidad:** SHAP values + Feature Importance

**Figuras generadas:**
- `fig09_target_regresion.png`
- `fig10_regresion_evaluacion.png`
- `fig11_regresion_comparativa.png`
- `fig12_shap_regresion.png`
- `fig13_importance_regresion.png`

**Outputs CSV:**
- `regresion_metricas.csv`
- `regresion_predicciones.csv`

---

### `04_no_supervisado.py`
**Objetivo:** Identificar perfiles de estudiantes mediante técnicas no supervisadas

**Técnicas aplicadas:**
1. **PCA** – Reducción de dimensionalidad y visualización
2. **K-Means** – Clustering con selección óptima de k (método del codo + Silhouette)
3. **Clustering Jerárquico** – Dendrograma (Ward)
4. **Análisis de estabilidad** – Bootstrap con 10 muestras

**Variables utilizadas:** Selección razonada de variables académicas y socioeconómicas (se excluyen macroeconómicas por ausencia de variabilidad individual)

**Figuras generadas:**
- `fig14_pca.png`
- `fig15_elbow_silhouette.png`
- `fig16_clusters_pca.png`
- `fig17_cluster_heatmap.png`
- `fig18_dendrograma.png`

**Outputs CSV:**
- `clusters_distribucion.csv`
- `clusters_perfiles.csv`

---

## Decisiones metodológicas clave

1. **División train/test:** 80/20 con estratificación por clase para mantener proporciones.
2. **Desbalanceo de clases:** SMOTE aplicado únicamente al conjunto de entrenamiento para evitar contaminación del test.
3. **Escalado:** StandardScaler ajustado en train y aplicado en test (pipeline correcto).
4. **Data leakage en regresión:** Variables del 2º semestre contemporáneas a la nota objetivo excluidas explícitamente.
5. **Selección de k en clustering:** Combinación de método del codo e índice Silhouette.
6. **Validación cruzada:** 5-fold en todos los modelos de regresión para estimación robusta.

---

## Reproducibilidad

Todos los modelos usan `random_state=42`. Los resultados son completamente reproducibles dado el dataset original.
