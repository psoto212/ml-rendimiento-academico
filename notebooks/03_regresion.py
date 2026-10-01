"""
=============================================================
PROYECTO FINAL - MACHINE LEARNING
Predicción del Éxito Académico en Educación Superior
=============================================================
Notebook 03: Tarea 2 – Regresión
Objetivo: predecir nota_media_2sem
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os, warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score)
import shap

DATA_PATH   = os.path.join(os.path.dirname(__file__), '..', 'data', 'rendimiento_estudiantes.csv')
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), '..', 'outputs')
os.makedirs(OUTPUT_PATH, exist_ok=True)

# ─── 1. Carga ────────────────────────────────────────────────────────────────
print("=" * 60)
print("1. CARGA Y PREPARACIÓN – REGRESIÓN")
print("=" * 60)

df = pd.read_csv(DATA_PATH, sep=';')

# ─── 2. Exclusión de variables por data leakage ─────────────────────────────
print("\n" + "=" * 60)
print("2. ANÁLISIS DE DATA LEAKAGE")
print("=" * 60)

TARGET = 'nota_media_2sem'

# Variables excluidas: contemporáneas al target
leakage_cols = [
    'asignaturas_2sem_matriculadas',    # cuántas se matriculó ese sem (simultánea al target)
    'asignaturas_2sem_evaluadas',       # cuántas fueron evaluadas (resultado del mismo sem)
    'asignaturas_2sem_aprobadas',       # directamente correlada con la nota
    'asignaturas_2sem_sin_evaluacion',  # información del mismo semestre
    'asignaturas_2sem_convalidadas',    # convalidaciones del 2º sem
    'objetivo',                         # variable target de clasificación (usa info final)
]

print("Variables excluidas por data leakage:")
for col in leakage_cols:
    print(f"  - {col}")
print("\nJustificación:")
print("  Estas variables son contemporáneas o equivalentes a nota_media_2sem.")
print("  Incluirlas implicaría que el modelo 've' información del futuro durante")
print("  la fase de entrenamiento, invalidando la predicción real.")
print("\nVariables PERMITIDAS del 2º semestre: ninguna (todas excluidas).")
print("El modelo predice a partir de: datos de acceso, contexto socioeconómico")
print("y rendimiento del 1er semestre.")

# ─── 3. Preparación de features ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. PREPARACIÓN DE FEATURES")
print("=" * 60)

df_reg = df.drop(columns=leakage_cols)

# Solo predecimos estudiantes con nota (nota 0 puede ser abandono temprano)
df_reg = df_reg[df_reg[TARGET] > 0].reset_index(drop=True)
print(f"Muestras con nota_media_2sem > 0: {len(df_reg)}")

y = df_reg[TARGET]
X_raw = df_reg.drop(columns=[TARGET])

print(f"\nEstadísticas del target:")
print(f"  Media:   {y.mean():.2f}")
print(f"  Mediana: {y.median():.2f}")
print(f"  Std:     {y.std():.2f}")
print(f"  Min-Max: {y.min():.2f} – {y.max():.2f}")

# Distribución del target
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(y, bins=40, color='#3498db', edgecolor='white', alpha=0.8)
axes[0].axvline(y.mean(), color='red', linestyle='--', label=f'Media={y.mean():.2f}')
axes[0].axvline(y.median(), color='orange', linestyle='--', label=f'Mediana={y.median():.2f}')
axes[0].set_title('Distribución de nota_media_2sem', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Nota Media 2º Semestre')
axes[0].legend()

axes[1].boxplot(y, patch_artist=True,
                boxprops={'facecolor': '#3498db', 'alpha': 0.7})
axes[1].set_title('Boxplot de nota_media_2sem', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Nota')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'fig09_target_regresion.png'), dpi=150, bbox_inches='tight')
plt.close()
print("→ fig09_target_regresion.png")

# ─── 4. Codificación ────────────────────────────────────────────────────────
bin_map = {'si': 1, 'no': 0, 'diurna': 1, 'vespertina': 0, 'hombre': 1, 'mujer': 0}
bin_cols = ['desplazado', 'necesidades_educativas_especiales', 'deudor',
            'matricula_al_dia', 'becado', 'internacional',
            'asistencia_diurna_vespertina', 'genero']

X_enc = X_raw.copy()
for col in bin_cols:
    if col in X_enc.columns:
        X_enc[col] = X_enc[col].map(bin_map).fillna(X_enc[col])

multi_cats = [c for c in X_enc.select_dtypes('object').columns]
X_enc = pd.get_dummies(X_enc, columns=multi_cats, drop_first=True)

print(f"\nFeatures finales: {X_enc.shape[1]}")

# ─── 5. Train/Test split y escalado ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_enc, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")

# ─── 6. Modelos ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. ENTRENAMIENTO Y EVALUACIÓN DE MODELOS")
print("=" * 60)

models = {
    'Regresión Lineal': LinearRegression(),
    'Ridge (α=1.0)':    Ridge(alpha=1.0),
    'Lasso (α=0.01)':   Lasso(alpha=0.01, max_iter=10000),
    'Árbol Decisión':   DecisionTreeRegressor(max_depth=6, min_samples_leaf=20, random_state=42),
    'Random Forest':    RandomForestRegressor(n_estimators=200, max_depth=12,
                                               min_samples_leaf=5, random_state=42, n_jobs=-1),
    'Gradient Boosting':GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                                   learning_rate=0.1, subsample=0.8, random_state=42),
}

reg_results = {}
for name, model in models.items():
    print(f"\n--- {name} ---")
    # CV
    cv_r2 = cross_val_score(model, X_train_sc, y_train, cv=5, scoring='r2')
    cv_rmse = cross_val_score(model, X_train_sc, y_train, cv=5,
                               scoring='neg_root_mean_squared_error')
    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)

    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    reg_results[name] = {
        'RMSE': rmse, 'MAE': mae, 'R²': r2,
        'CV R² (mean)': cv_r2.mean(), 'CV R² (std)': cv_r2.std(),
        'y_pred': y_pred, 'model': model
    }

    print(f"  RMSE:        {rmse:.4f}")
    print(f"  MAE:         {mae:.4f}")
    print(f"  R²:          {r2:.4f}")
    print(f"  CV R² (5-fold): {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")

# ─── 7. Tabla comparativa ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. TABLA COMPARATIVA")
print("=" * 60)

df_reg_res = pd.DataFrame(
    {k: {m: v for m, v in d.items() if m not in ['y_pred', 'model']}
     for k, d in reg_results.items()}
).T
print(df_reg_res.round(4).to_string())
df_reg_res.round(4).to_csv(os.path.join(OUTPUT_PATH, 'regresion_metricas.csv'))

# ─── 8. Figuras de evaluación ────────────────────────────────────────────────
best_reg_name = max(reg_results, key=lambda k: reg_results[k]['R²'])
print(f"\nMejor modelo de regresión: {best_reg_name}")

best_pred = reg_results[best_reg_name]['y_pred']
best_model = reg_results[best_reg_name]['model']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Predicted vs Real
axes[0].scatter(y_test, best_pred, alpha=0.4, color='#3498db', s=20)
lims = [min(y_test.min(), best_pred.min()), max(y_test.max(), best_pred.max())]
axes[0].plot(lims, lims, 'r--', linewidth=1.5, label='Predicción perfecta')
axes[0].set_xlabel('Valor Real')
axes[0].set_ylabel('Valor Predicho')
axes[0].set_title(f'Predichos vs Reales\n{best_reg_name}', fontsize=11, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Residuos
residuals = y_test.values - best_pred
axes[1].hist(residuals, bins=40, color='#e74c3c', edgecolor='white', alpha=0.8)
axes[1].axvline(0, color='black', linestyle='--', linewidth=1.5)
axes[1].set_xlabel('Residuo (Real – Predicho)')
axes[1].set_ylabel('Frecuencia')
axes[1].set_title('Distribución de Residuos', fontsize=11, fontweight='bold')
axes[1].grid(alpha=0.3)

plt.suptitle(f'Evaluación del Modelo de Regresión ({best_reg_name})', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'fig10_regresion_evaluacion.png'), dpi=150, bbox_inches='tight')
plt.close()
print("→ fig10_regresion_evaluacion.png")

# ─── 9. Comparativa de modelos lineales vs no lineales ──────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
names = list(reg_results.keys())
r2_vals   = [reg_results[n]['R²'] for n in names]
rmse_vals = [reg_results[n]['RMSE'] for n in names]
colors_r = ['#3498db', '#3498db', '#3498db', '#e74c3c', '#e74c3c', '#e74c3c']

axes[0].barh(names, r2_vals, color=colors_r, edgecolor='white')
axes[0].set_xlabel('R²')
axes[0].set_title('R² por Modelo', fontweight='bold')
axes[0].axvline(0, color='gray', linewidth=0.8)
for i, v in enumerate(r2_vals):
    axes[0].text(max(0, v) + 0.005, i, f'{v:.3f}', va='center', fontsize=9)

axes[1].barh(names, rmse_vals, color=colors_r, edgecolor='white')
axes[1].set_xlabel('RMSE')
axes[1].set_title('RMSE por Modelo', fontweight='bold')
for i, v in enumerate(rmse_vals):
    axes[1].text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)

from matplotlib.patches import Patch
legend = [Patch(color='#3498db', label='Modelos Lineales'),
          Patch(color='#e74c3c', label='Modelos No Lineales')]
fig.legend(handles=legend, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.05))
plt.suptitle('Comparativa Lineal vs No Lineal', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'fig11_regresion_comparativa.png'), dpi=150, bbox_inches='tight')
plt.close()
print("→ fig11_regresion_comparativa.png")

# ─── 10. SHAP values ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. INTERPRETABILIDAD – SHAP VALUES")
print("=" * 60)

try:
    rf_reg = reg_results['Random Forest']['model']
    explainer = shap.TreeExplainer(rf_reg)
    # Muestra aleatoria para SHAP
    np.random.seed(42)
    idx_sample = np.random.choice(len(X_test_sc), size=min(200, len(X_test_sc)), replace=False)
    X_sample = X_test_sc[idx_sample]

    shap_values = explainer.shap_values(X_sample)

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(shap_values, X_sample,
                      feature_names=X_enc.columns.tolist(),
                      plot_type='bar', show=False, max_display=15)
    plt.title('SHAP – Importancia de Variables (Random Forest Regresión)',
              fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'fig12_shap_regresion.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("→ fig12_shap_regresion.png")
except Exception as e:
    print(f"  SHAP no disponible: {e}")

# ─── 11. Feature importance RF regresión ────────────────────────────────────
rf_reg_model = reg_results['Random Forest']['model']
imp_reg = pd.Series(rf_reg_model.feature_importances_, index=X_enc.columns).nlargest(15)

fig, ax = plt.subplots(figsize=(10, 6))
imp_reg[::-1].plot(kind='barh', ax=ax, color='#9b59b6', edgecolor='white')
ax.set_title('Top 15 Variables – Random Forest Regresión\n(Feature Importance)',
             fontsize=12, fontweight='bold')
ax.set_xlabel('Importancia')
ax.grid(axis='x', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'fig13_importance_regresion.png'), dpi=150, bbox_inches='tight')
plt.close()
print("→ fig13_importance_regresion.png")

print(f"\nTop 10 variables para predecir nota_media_2sem:")
print(imp_reg.head(10).round(4).to_string())

# ─── 12. Guardar predicciones ────────────────────────────────────────────────
pred_reg_df = pd.DataFrame({
    'real_nota_2sem': y_test.values,
    'pred_RF':  reg_results['Random Forest']['y_pred'],
    'pred_GB':  reg_results['Gradient Boosting']['y_pred'],
    'pred_Ridge': reg_results['Ridge (α=1.0)']['y_pred'],
})
pred_reg_df.to_csv(os.path.join(OUTPUT_PATH, 'regresion_predicciones.csv'), index=False)
print("\n→ regresion_predicciones.csv guardado.")

print("\n✓ Tarea 2 (Regresión) completada.")
