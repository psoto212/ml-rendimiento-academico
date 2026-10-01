"""
=============================================================
PROYECTO FINAL - MACHINE LEARNING
Predicción del Éxito Académico en Educación Superior
=============================================================
Notebook 01: Exploración y Análisis Descriptivo (EDA) + Preprocesado
"""

# ─── Librerías ──────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings, os
warnings.filterwarnings('ignore')

# ─── Rutas ──────────────────────────────────────────────────────────────────
DATA_PATH   = os.path.join(os.path.dirname(__file__), '..', 'data', 'rendimiento_estudiantes.csv')
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), '..', 'outputs')
os.makedirs(OUTPUT_PATH, exist_ok=True)

# ─── 1. Carga de datos ──────────────────────────────────────────────────────
print("=" * 60)
print("1. CARGA Y VISIÓN GENERAL DEL DATASET")
print("=" * 60)

df = pd.read_csv(DATA_PATH, sep=';')
print(f"Dimensiones: {df.shape[0]} filas × {df.shape[1]} columnas")
print(f"\nPrimeras filas:\n{df.head(3).to_string()}")

# ─── 2. Estadísticas descriptivas ───────────────────────────────────────────
print("\n" + "=" * 60)
print("2. ESTADÍSTICAS DESCRIPTIVAS")
print("=" * 60)

num_cols = df.select_dtypes(include='number').columns.tolist()
cat_cols = df.select_dtypes(include='object').columns.tolist()
cat_cols_feat = [c for c in cat_cols if c != 'objetivo']

print(f"\nVariables numéricas ({len(num_cols)}): {num_cols}")
print(f"\nVariables categóricas ({len(cat_cols)}): {cat_cols}")
print(f"\nEstadísticas numéricas:\n{df[num_cols].describe().round(2).to_string()}")

# Valores nulos
print(f"\nValores nulos por columna:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
print(f"Total nulos: {df.isnull().sum().sum()}")

# ─── 3. Variable objetivo ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. DISTRIBUCIÓN DE LA VARIABLE OBJETIVO")
print("=" * 60)

target_counts = df['objetivo'].value_counts()
target_pct    = df['objetivo'].value_counts(normalize=True) * 100
print(target_counts.to_string())
print(f"\nPorcentajes:\n{target_pct.round(1).to_string()}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
colors = ['red', 'blue', 'green']
# Barras
axes[0].bar(target_counts.index, target_counts.values, color=colors, edgecolor='white', linewidth=1.5)
axes[0].set_title('Distribución de la Variable Objetivo', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Estado del Estudiante')
axes[0].set_ylabel('Número de Estudiantes')
for i, (v, p) in enumerate(zip(target_counts.values, target_pct.values)):
    axes[0].text(i, v + 20, f'{v}\n({p:.1f}%)', ha='center', fontsize=10)

# Tarta
axes[1].pie(target_counts.values, labels=target_counts.index, autopct='%1.1f%%',
            colors=colors, startangle=90, textprops={'fontsize': 11})
axes[1].set_title('Proporción por Clase', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'fig01_distribucion_objetivo.png'), dpi=150, bbox_inches='tight')
plt.close()
print("→ Figura guardada: fig01_distribucion_objetivo.png")

# ─── 4. Análisis univariante ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. ANÁLISIS UNIVARIANTE - VARIABLES NUMÉRICAS")
print("=" * 60)

fig, axes = plt.subplots(3, 4, figsize=(18, 12))
axes = axes.flatten()
palette = {'abandono': 'red', 'matriculado': 'blue', 'graduado': 'green'}
key_num = ['edad_al_matricularse', 'nota_admision', 'nota_cualificacion_previa',
           'nota_media_1sem', 'nota_media_2sem', 'asignaturas_1sem_aprobadas',
           'asignaturas_2sem_aprobadas', 'asignaturas_1sem_matriculadas',
           'asignaturas_1sem_evaluadas', 'tasa_desempleo', 'tasa_inflacion', 'pib']

for i, col in enumerate(key_num):
    for clase, color in palette.items():
        datos = df[df['objetivo'] == clase][col].dropna()
        axes[i].hist(datos, bins=25, alpha=0.5, color=color, label=clase, density=True)
    axes[i].set_title(col.replace('_', ' ').title(), fontsize=9)
    axes[i].legend(fontsize=7)
    axes[i].tick_params(labelsize=7)

plt.suptitle('Distribución de Variables Numéricas por Clase', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'fig02_univariante_numericas.png'), dpi=150, bbox_inches='tight')
plt.close()
print("→ Figura guardada: fig02_univariante_numericas.png")

# ─── 5. Correlación ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. MATRIZ DE CORRELACIÓN")
print("=" * 60)

corr = df[num_cols].corr()
fig, ax = plt.subplots(figsize=(13, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            linewidths=0.5, ax=ax, annot_kws={'size': 8})
ax.set_title('Matriz de Correlación – Variables Numéricas', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'fig03_correlacion.png'), dpi=150, bbox_inches='tight')
plt.close()
print("→ Figura guardada: fig03_correlacion.png")

# ─── 6. Análisis bivariante: notas por clase ────────────────────────────────
print("\n" + "=" * 60)
print("6. ANÁLISIS BIVARIANTE")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
key_vars = ['nota_admision', 'nota_media_1sem', 'nota_media_2sem']
titles   = ['Nota de Admisión', 'Nota Media 1er Sem.', 'Nota Media 2º Sem.']

for ax, var, title in zip(axes, key_vars, titles):
    data_plot = [df[df['objetivo'] == c][var].dropna().values for c in ['abandono', 'matriculado', 'graduado']]
    bp = ax.boxplot(data_plot, labels=['Abandono', 'Matriculado', 'Graduado'],
                    patch_artist=True, notch=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylabel('Puntuación')
    ax.grid(axis='y', alpha=0.4)

plt.suptitle('Distribución de Notas por Estado del Estudiante', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'fig04_notas_por_clase.png'), dpi=150, bbox_inches='tight')
plt.close()
print("→ Figura guardada: fig04_notas_por_clase.png")

# Becados por clase
beca_clase = pd.crosstab(df['becado'], df['objetivo'], normalize='columns') * 100
print(f"\nPorcentaje de becados por clase:\n{beca_clase.round(1).to_string()}")

# Deudores por clase
deuda_clase = pd.crosstab(df['deudor'], df['objetivo'], normalize='columns') * 100
print(f"\nPorcentaje de deudores por clase:\n{deuda_clase.round(1).to_string()}")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
cat_interest = ['becado', 'deudor', 'matricula_al_dia']
for ax, var in zip(axes, cat_interest):
    ct = pd.crosstab(df[var], df['objetivo'], normalize='columns') * 100
    ct.plot(kind='bar', ax=ax, color=colors, edgecolor='white')
    ax.set_title(var.replace('_', ' ').title(), fontsize=11, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('Porcentaje (%)')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend(title='Estado', fontsize=8)
    ax.grid(axis='y', alpha=0.4)

plt.suptitle('Variables Socioeconómicas por Estado del Estudiante', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'fig05_variables_socioeconomicas.png'), dpi=150, bbox_inches='tight')
plt.close()
print("→ Figura guardada: fig05_variables_socioeconomicas.png")

# ─── 7. Preprocesado ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("7. PREPROCESADO Y EXPORTACIÓN")
print("=" * 60)

# Columnas del 2º semestre (todas)
cols_2sem = [c for c in df.columns if '2sem' in c]
print(f"Columnas 2º semestre: {cols_2sem}")

# Variables a excluir en REGRESIÓN (data leakage)
leakage_cols = ['asignaturas_2sem_matriculadas', 'asignaturas_2sem_evaluadas',
                'asignaturas_2sem_aprobadas', 'asignaturas_2sem_sin_evaluacion',
                'asignaturas_2sem_convalidadas']
print(f"\nVariables excluidas por leakage en regresión:\n{leakage_cols}")
print("Justificación: son contemporáneas a nota_media_2sem (target de regresión).")

# Codificación de categóricas
df_proc = df.copy()
cat_feat = [c for c in df.select_dtypes('object').columns if c != 'objetivo']

# Label encoding para binarias simples
bin_cols = ['desplazado', 'necesidades_educativas_especiales', 'deudor',
            'matricula_al_dia', 'becado', 'internacional',
            'asistencia_diurna_vespertina', 'genero']

for col in bin_cols:
    if col in df_proc.columns:
        df_proc[col] = df_proc[col].map(lambda x: 1 if x in ['si', 'diurna', 'hombre'] else 0)

# One-hot encoding para el resto de categóricas
multi_cats = [c for c in cat_feat if c not in bin_cols]
df_proc = pd.get_dummies(df_proc, columns=multi_cats, drop_first=True)

# Encode objetivo
label_map = {'abandono': 0, 'matriculado': 1, 'graduado': 2}
df_proc['objetivo_enc'] = df_proc['objetivo'].map(label_map)

print(f"\nShape tras preprocesado: {df_proc.shape}")

# Guardar
df_proc.to_csv(os.path.join(OUTPUT_PATH, 'dataset_procesado.csv'), index=False)
print("→ Dataset procesado guardado: outputs/dataset_procesado.csv")

print("\n EDA y preprocesado completados.")
