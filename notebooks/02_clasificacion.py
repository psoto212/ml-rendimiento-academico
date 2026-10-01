"""
=============================================================
PROYECTO FINAL - MACHINE LEARNING
Predicción del Éxito Académico en Educación Superior
=============================================================
Notebook 02: Tarea 1 – Clasificación Multiclase
Objetivo: predecir estado del estudiante (Abandono / Matriculado / Graduado)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os, warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix,
                              f1_score, accuracy_score, roc_auc_score,
                              ConfusionMatrixDisplay)
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

DATA_PATH   = os.path.join(os.path.dirname(__file__), '..', 'data', 'rendimiento_estudiantes.csv')
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), '..', 'outputs')
os.makedirs(OUTPUT_PATH, exist_ok=True)

# ─── 1. Carga y preprocesado inline ─────────────────────────────────────────
print("=" * 60)
print("1. CARGA Y PREPARACIÓN DE DATOS")
print("=" * 60)

df = pd.read_csv(DATA_PATH, sep=';')

# Codificación binaria
bin_map = {'si': 1, 'no': 0, 'diurna': 1, 'vespertina': 0, 'hombre': 1, 'mujer': 0}
bin_cols = ['desplazado', 'necesidades_educativas_especiales', 'deudor',
            'matricula_al_dia', 'becado', 'internacional',
            'asistencia_diurna_vespertina', 'genero']

df_enc = df.copy()
for col in bin_cols:
    df_enc[col] = df_enc[col].map(bin_map).fillna(df_enc[col])

# One-hot para el resto de categóricas (excepto objetivo)
multi_cats = [c for c in df_enc.select_dtypes('object').columns if c != 'objetivo']
df_enc = pd.get_dummies(df_enc, columns=multi_cats, drop_first=True)

# Target
label_map = {'abandono': 0, 'matriculado': 1, 'graduado': 2}
inv_label  = {v: k for k, v in label_map.items()}
y = df_enc['objetivo'].map(label_map)
X = df_enc.drop(columns=['objetivo'])

print(f"Features: {X.shape[1]}  |  Muestras: {X.shape[0]}")
print(f"Distribución objetivo:\n{y.value_counts().rename(inv_label)}")

# ─── 2. División train / test estratificada ─────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
print(f"\nTrain: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")

# ─── 3. Escalado ────────────────────────────────────────────────────────────
scaler  = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ─── 4. Manejo del desbalanceo: SMOTE en train ──────────────────────────────
print("\n" + "=" * 60)
print("2. MANEJO DEL DESBALANCEO DE CLASES")
print("=" * 60)
print("Distribución original en train:")
print(pd.Series(y_train).value_counts().rename(inv_label))

smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_sc, y_train)
print("\nDistribución tras SMOTE:")
print(pd.Series(y_train_bal).value_counts().rename(inv_label))

# ─── 5. Definición de modelos ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. ENTRENAMIENTO Y EVALUACIÓN DE MODELOS")
print("=" * 60)

models = {
    'Regresión Logística': LogisticRegression(
        max_iter=1000, solver='lbfgs', random_state=42),
    'Árbol de Decisión': DecisionTreeClassifier(
        max_depth=8, min_samples_leaf=10, random_state=42),
    'Random Forest': RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_leaf=5,
        class_weight='balanced', random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        subsample=0.8, random_state=42),
}

results = {}
for name, model in models.items():
    print(f"\n--- {name} ---")
    model.fit(X_train_bal, y_train_bal)
    y_pred = model.predict(X_test_sc)

    acc  = accuracy_score(y_test, y_pred)
    f1_w = f1_score(y_test, y_pred, average='weighted')
    f1_m = f1_score(y_test, y_pred, average='macro')

    results[name] = {'Accuracy': acc, 'F1 Weighted': f1_w, 'F1 Macro': f1_m,
                     'y_pred': y_pred, 'model': model}

    print(f"  Accuracy:    {acc:.4f}")
    print(f"  F1 Weighted: {f1_w:.4f}")
    print(f"  F1 Macro:    {f1_m:.4f}")
    print(classification_report(y_test, y_pred,
                                  target_names=['Abandono', 'Matriculado', 'Graduado']))

# ─── 6. Tabla comparativa ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. TABLA COMPARATIVA DE MODELOS")
print("=" * 60)

df_results = pd.DataFrame(
    {k: {m: v for m, v in d.items() if m not in ['y_pred', 'model']}
     for k, d in results.items()}
).T
print(df_results.round(4).to_string())
df_results.round(4).to_csv(os.path.join(OUTPUT_PATH, 'clasificacion_metricas.csv'))

# ─── 7. Figura: métricas comparativas ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(results))
width = 0.25
colors_bar = ['#3498db', '#e74c3c', '#2ecc71']

for i, metric in enumerate(['Accuracy', 'F1 Weighted', 'F1 Macro']):
    vals = [results[m][metric] for m in results]
    bars = ax.bar(x + i * width, vals, width, label=metric,
                  color=colors_bar[i], alpha=0.85, edgecolor='white')

ax.set_xticks(x + width)
ax.set_xticklabels(list(results.keys()), fontsize=10)
ax.set_ylim(0.5, 1.0)
ax.set_ylabel('Puntuación')
ax.set_title('Comparativa de Modelos de Clasificación', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'fig06_clasificacion_metricas.png'), dpi=150, bbox_inches='tight')
plt.close()
print("→ fig06_clasificacion_metricas.png")

# ─── 8. Matriz de confusión del mejor modelo ────────────────────────────────
best_name = max(results, key=lambda k: results[k]['F1 Weighted'])
print(f"\nMejor modelo: {best_name}")
best_pred = results[best_name]['y_pred']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Confusión absoluta
cm = confusion_matrix(y_test, best_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=['Abandono', 'Matriculado', 'Graduado'])
disp.plot(ax=axes[0], cmap='Blues', colorbar=False)
axes[0].set_title(f'Matriz de Confusión – {best_name}\n(valores absolutos)', fontsize=11)

# Confusión normalizada
cm_norm = confusion_matrix(y_test, best_pred, normalize='true')
disp2 = ConfusionMatrixDisplay(cm_norm.round(2), display_labels=['Abandono', 'Matriculado', 'Graduado'])
disp2.plot(ax=axes[1], cmap='Blues', colorbar=False)
axes[1].set_title(f'Matriz de Confusión – {best_name}\n(normalizada por fila)', fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'fig07_confusion_matrix.png'), dpi=150, bbox_inches='tight')
plt.close()
print("→ fig07_confusion_matrix.png")

# ─── 9. Importancia de variables (Random Forest) ────────────────────────────
print("\n" + "=" * 60)
print("5. IMPORTANCIA DE VARIABLES – RANDOM FOREST")
print("=" * 60)

rf_model = results['Random Forest']['model']
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
top20 = importances.nlargest(20)

fig, ax = plt.subplots(figsize=(10, 7))
colors_imp = ['#e74c3c' if i < 5 else '#3498db' for i in range(20)]
top20[::-1].plot(kind='barh', ax=ax, color=colors_imp[::-1], edgecolor='white')
ax.set_title('Top 20 Variables más Importantes\n(Random Forest – Feature Importance)',
             fontsize=12, fontweight='bold')
ax.set_xlabel('Importancia (Gini)')
ax.grid(axis='x', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'fig08_feature_importance_clf.png'), dpi=150, bbox_inches='tight')
plt.close()
print("→ fig08_feature_importance_clf.png")

print(f"\nTop 10 variables más importantes:")
print(top20.head(10).round(4).to_string())

# ─── 10. Análisis del efecto del desbalanceo ────────────────────────────────
print("\n" + "=" * 60)
print("6. ANÁLISIS DEL DESBALANCEO DE CLASES")
print("=" * 60)

# Comparación: RF sin SMOTE vs con SMOTE
rf_nobal = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf_nobal.fit(X_train_sc, y_train)
pred_nobal = rf_nobal.predict(X_test_sc)

rf_bal = results['Random Forest']['model']
pred_bal = rf_bal.predict(X_test_sc)

print("Sin SMOTE:")
print(f"  F1 Macro:    {f1_score(y_test, pred_nobal, average='macro'):.4f}")
print(f"  F1 Weighted: {f1_score(y_test, pred_nobal, average='weighted'):.4f}")
print(f"  F1 Matriculado: {f1_score(y_test, pred_nobal, average=None)[1]:.4f}")

print("\nCon SMOTE (class_weight=balanced):")
print(f"  F1 Macro:    {f1_score(y_test, pred_bal, average='macro'):.4f}")
print(f"  F1 Weighted: {f1_score(y_test, pred_bal, average='weighted'):.4f}")
print(f"  F1 Matriculado: {f1_score(y_test, pred_bal, average=None)[1]:.4f}")

# ─── 11. Guardar predicciones ────────────────────────────────────────────────
pred_df = pd.DataFrame({
    'real': y_test.map(inv_label).values,
    'prediccion_RF': pd.Series(pred_bal).map(inv_label).values,
    'prediccion_LR': pd.Series(results['Regresión Logística']['y_pred']).map(inv_label).values,
    'prediccion_GB': pd.Series(results['Gradient Boosting']['y_pred']).map(inv_label).values,
})
pred_df.to_csv(os.path.join(OUTPUT_PATH, 'clasificacion_predicciones.csv'), index=False)
print("\n→ clasificacion_predicciones.csv guardado.")

print("\n✓ Tarea 1 (Clasificación) completada.")
