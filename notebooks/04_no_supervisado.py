"""
=============================================================
PROYECTO FINAL - MACHINE LEARNING
Predicción del Éxito Académico en Educación Superior
=============================================================
Notebook 04: Tarea 3 – Aprendizaje No Supervisado
Técnicas: PCA + K-Means + análisis de clusters
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os, warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage

DATA_PATH   = os.path.join(os.path.dirname(__file__), '..', 'data', 'rendimiento_estudiantes.csv')
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), '..', 'outputs')
os.makedirs(OUTPUT_PATH, exist_ok=True)

# ─── 1. Carga y selección de variables ──────────────────────────────────────
print("=" * 60)
print("1. CARGA Y SELECCIÓN DE VARIABLES")
print("=" * 60)

df = pd.read_csv(DATA_PATH, sep=';')

# Selección de variables relevantes para perfilado de estudiantes
# Se excluyen variables macroeconómicas (igual para todos en mismo período)
# y se incluyen variables de perfil académico y socioeconómico

num_vars = [
    'edad_al_matricularse', 'nota_admision', 'nota_cualificacion_previa',
    'asignaturas_1sem_matriculadas', 'asignaturas_1sem_evaluadas',
    'asignaturas_1sem_aprobadas', 'nota_media_1sem',
    'asignaturas_2sem_matriculadas', 'asignaturas_2sem_aprobadas', 'nota_media_2sem',
]

bin_vars = ['becado', 'deudor', 'matricula_al_dia', 'internacional',
            'desplazado', 'genero']

print(f"Variables numéricas seleccionadas: {num_vars}")
print(f"Variables binarias seleccionadas:  {bin_vars}")
print("\nJustificación: Se elige un subconjunto interpretable que capture perfil")
print("académico (notas, asignaturas), socioeconómico (beca, deuda) y demográfico.")
print("Variables macroeconómicas excluidas (sin variabilidad entre estudiantes).")

# Codificación
bin_map = {'si': 1, 'no': 0, 'hombre': 1, 'mujer': 0}
df_uns = df[num_vars + bin_vars + ['objetivo']].copy()
for col in bin_vars:
    df_uns[col] = df_uns[col].map(bin_map)

df_uns = df_uns.dropna().reset_index(drop=True)
print(f"\nMuestras disponibles: {len(df_uns)}")

X_uns = df_uns[num_vars + bin_vars]
y_true = df_uns['objetivo']
label_enc = {'abandono': 0, 'matriculado': 1, 'graduado': 2}
y_enc = y_true.map(label_enc)

# ─── 2. Escalado y PCA ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. REDUCCIÓN DE DIMENSIONALIDAD – PCA")
print("=" * 60)

scaler = StandardScaler()
X_sc = scaler.fit_transform(X_uns)

pca_full = PCA(random_state=42)
pca_full.fit(X_sc)

# Varianza explicada acumulada
exp_var = pca_full.explained_variance_ratio_
cum_var = np.cumsum(exp_var)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].bar(range(1, len(exp_var)+1), exp_var * 100, color='#3498db', edgecolor='white', alpha=0.8)
axes[0].step(range(1, len(cum_var)+1), cum_var * 100, color='red', linewidth=2, where='mid', label='Acumulada')
axes[0].axhline(80, color='orange', linestyle='--', label='80% umbral')
axes[0].axhline(90, color='green', linestyle='--', label='90% umbral')
axes[0].set_xlabel('Componente Principal')
axes[0].set_ylabel('Varianza Explicada (%)')
axes[0].set_title('Varianza Explicada por Componente', fontsize=11, fontweight='bold')
axes[0].legend()
axes[0].set_xlim(0.5, min(20, len(exp_var)) + 0.5)

# Biplot PCA (PC1 vs PC2)
pca2 = PCA(n_components=2, random_state=42)
X_pca2 = pca2.fit_transform(X_sc)

colors_obj = {'abandono': '#e74c3c', 'matriculado': '#3498db', 'graduado': '#2ecc71'}
for clase, color in colors_obj.items():
    mask = y_true == clase
    axes[1].scatter(X_pca2[mask, 0], X_pca2[mask, 1],
                    c=color, label=clase, alpha=0.4, s=15)
axes[1].set_xlabel(f'PC1 ({exp_var[0]*100:.1f}%)')
axes[1].set_ylabel(f'PC2 ({exp_var[1]*100:.1f}%)')
axes[1].set_title('PCA – Proyección 2D por Clase Real', fontsize=11, fontweight='bold')
axes[1].legend(markerscale=2)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'fig14_pca.png'), dpi=150, bbox_inches='tight')
plt.close()
print("→ fig14_pca.png")

n_comp_90 = np.argmax(cum_var >= 0.90) + 1
print(f"\nComponentes necesarios para 90% varianza: {n_comp_90}")
print(f"PC1 varianza: {exp_var[0]*100:.1f}%  |  PC2 varianza: {exp_var[1]*100:.1f}%")

# Loadings de PC1 y PC2
feat_names = num_vars + bin_vars
pca_loadings = pd.DataFrame(pca2.components_.T, index=feat_names, columns=['PC1', 'PC2'])
print(f"\nLoadings PC1 (mayores pesos):\n{pca_loadings['PC1'].abs().nlargest(5)}")
print(f"\nLoadings PC2 (mayores pesos):\n{pca_loadings['PC2'].abs().nlargest(5)}")

# ─── 3. Clustering K-Means ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. K-MEANS CLUSTERING")
print("=" * 60)

# Elbow + Silhouette para elegir k
inertias   = []
silhouettes = []
k_range = range(2, 9)

pca_n = PCA(n_components=n_comp_90, random_state=42)
X_pcaN = pca_n.fit_transform(X_sc)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_pcaN)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_pcaN, labels))
    print(f"  k={k}: Inercia={km.inertia_:.1f}, Silhouette={silhouettes[-1]:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(list(k_range), inertias, 'o-', color='#3498db', linewidth=2, markersize=8)
axes[0].set_xlabel('Número de Clusters (k)')
axes[0].set_ylabel('Inercia (Within-cluster SSE)')
axes[0].set_title('Método del Codo – K-Means', fontsize=11, fontweight='bold')
axes[0].grid(alpha=0.4)

axes[1].plot(list(k_range), silhouettes, 's-', color='#e74c3c', linewidth=2, markersize=8)
axes[1].set_xlabel('Número de Clusters (k)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Score por k', fontsize=11, fontweight='bold')
axes[1].grid(alpha=0.4)
axes[1].axvline(silhouettes.index(max(silhouettes)) + 2, color='green',
                linestyle='--', label=f'Óptimo k={silhouettes.index(max(silhouettes))+2}')
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'fig15_elbow_silhouette.png'), dpi=150, bbox_inches='tight')
plt.close()
print("→ fig15_elbow_silhouette.png")

# Mejor k
best_k = silhouettes.index(max(silhouettes)) + 2
print(f"\nK óptimo según Silhouette: {best_k}")

# ─── 4. Clustering final con k óptimo ───────────────────────────────────────
print("\n" + "=" * 60)
print(f"4. K-MEANS FINAL (k={best_k})")
print("=" * 60)

km_final = KMeans(n_clusters=best_k, random_state=42, n_init=20)
cluster_labels = km_final.fit_predict(X_pcaN)

sil_final = silhouette_score(X_pcaN, cluster_labels)
db_final  = davies_bouldin_score(X_pcaN, cluster_labels)
print(f"Silhouette Score: {sil_final:.4f}")
print(f"Davies-Bouldin:   {db_final:.4f}")

df_uns['cluster'] = cluster_labels

# Distribución objetivo por cluster
ct = pd.crosstab(df_uns['cluster'], y_true, normalize='index') * 100
print(f"\nDistribución de clases reales por cluster (%):\n{ct.round(1).to_string()}")

ct.to_csv(os.path.join(OUTPUT_PATH, 'clusters_distribucion.csv'))

# ─── 5. Visualización clusters ──────────────────────────────────────────────
colors_k = plt.cm.Set2(np.linspace(0, 1, best_k))

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Clusters en PCA 2D
for k in range(best_k):
    mask = cluster_labels == k
    axes[0].scatter(X_pca2[mask, 0], X_pca2[mask, 1],
                    c=[colors_k[k]], label=f'Cluster {k}', alpha=0.5, s=20)

axes[0].set_xlabel(f'PC1 ({exp_var[0]*100:.1f}%)')
axes[0].set_ylabel(f'PC2 ({exp_var[1]*100:.1f}%)')
axes[0].set_title(f'K-Means (k={best_k}) en espacio PCA', fontsize=11, fontweight='bold')
axes[0].legend(markerscale=2)
axes[0].grid(alpha=0.3)

# Composición de cada cluster
ct_abs = pd.crosstab(df_uns['cluster'], y_true)
ct_abs.plot(kind='bar', ax=axes[1], color=['#e74c3c', '#3498db', '#2ecc71'],
            edgecolor='white', stacked=True)
axes[1].set_title('Composición de cada Cluster\n(por estado real del estudiante)',
                   fontsize=11, fontweight='bold')
axes[1].set_xlabel('Cluster')
axes[1].set_ylabel('Número de Estudiantes')
axes[1].tick_params(axis='x', rotation=0)
axes[1].legend(title='Estado', bbox_to_anchor=(1, 1))

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'fig16_clusters_pca.png'), dpi=150, bbox_inches='tight')
plt.close()
print("→ fig16_clusters_pca.png")

# ─── 6. Perfil de cada cluster ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. PERFIL DE CADA CLUSTER")
print("=" * 60)

cluster_profile = df_uns.groupby('cluster')[num_vars + bin_vars].mean().round(3)
print(cluster_profile.T.to_string())
cluster_profile.T.to_csv(os.path.join(OUTPUT_PATH, 'clusters_perfiles.csv'))

# Heatmap de perfiles
fig, ax = plt.subplots(figsize=(max(6, best_k * 2), 9))
profile_norm = (cluster_profile.T - cluster_profile.T.min(axis=1).values.reshape(-1,1)) / \
               (cluster_profile.T.max(axis=1) - cluster_profile.T.min(axis=1)).values.reshape(-1,1)
profile_norm = profile_norm.fillna(0)

sns.heatmap(profile_norm, annot=cluster_profile.T.round(2), fmt='g',
            cmap='RdYlGn', ax=ax, linewidths=0.5,
            cbar_kws={'label': 'Valor normalizado [0-1]'})
ax.set_title('Perfil Medio por Cluster (heatmap normalizado)', fontsize=12, fontweight='bold')
ax.set_xlabel('Cluster')
ax.set_ylabel('Variable')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'fig17_cluster_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()
print("→ fig17_cluster_heatmap.png")

# ─── 7. Dendrograma jerárquico ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. CLUSTERING JERÁRQUICO – DENDROGRAMA")
print("=" * 60)

# Muestra aleatoria para el dendrograma
np.random.seed(42)
idx_dend = np.random.choice(len(X_pcaN), size=min(300, len(X_pcaN)), replace=False)
X_dend = X_pcaN[idx_dend]

Z = linkage(X_dend, method='ward')

fig, ax = plt.subplots(figsize=(14, 5))
dendrogram(Z, ax=ax, truncate_mode='lastp', p=30,
           leaf_rotation=45, leaf_font_size=9, color_threshold=0)
ax.set_title('Dendrograma Jerárquico (Ward, muestra n=300)', fontsize=12, fontweight='bold')
ax.set_xlabel('Índice de muestra')
ax.set_ylabel('Distancia')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'fig18_dendrograma.png'), dpi=150, bbox_inches='tight')
plt.close()
print("→ fig18_dendrograma.png")

# ─── 8. Estabilidad de clusters ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("7. ESTABILIDAD DE CLUSTERS (BOOTSTRAP)")
print("=" * 60)

sil_scores = []
for seed in range(10):
    np.random.seed(seed)
    idx = np.random.choice(len(X_pcaN), size=int(0.8 * len(X_pcaN)), replace=False)
    km_s = KMeans(n_clusters=best_k, random_state=seed, n_init=10)
    lbl_s = km_s.fit_predict(X_pcaN[idx])
    sil_scores.append(silhouette_score(X_pcaN[idx], lbl_s))

print(f"Silhouette en 10 muestras bootstrap (80%):")
print(f"  Media:  {np.mean(sil_scores):.4f}")
print(f"  Std:    {np.std(sil_scores):.4f}")
print(f"  Min:    {np.min(sil_scores):.4f}")
print(f"  Max:    {np.max(sil_scores):.4f}")
print(f"  Valores: {[round(s,4) for s in sil_scores]}")

# ─── 9. Resumen interpretativo ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("8. INTERPRETACIÓN DE PERFILES")
print("=" * 60)

for k in range(best_k):
    subset = df_uns[df_uns['cluster'] == k]
    dominant = y_true[df_uns['cluster'] == k].value_counts().idxmax()
    pct = (y_true[df_uns['cluster'] == k].value_counts().max() / len(subset) * 100)
    print(f"\n=== Cluster {k} ({len(subset)} estudiantes, {pct:.0f}% {dominant}) ===")
    print(f"  Nota 1sem:       {subset['nota_media_1sem'].mean():.2f}")
    print(f"  Nota 2sem:       {subset['nota_media_2sem'].mean():.2f}")
    print(f"  Aprobadas 1sem:  {subset['asignaturas_1sem_aprobadas'].mean():.1f}")
    print(f"  Aprobadas 2sem:  {subset['asignaturas_2sem_aprobadas'].mean():.1f}")
    print(f"  Becado:          {subset['becado'].mean()*100:.0f}%")
    print(f"  Deudor:          {subset['deudor'].mean()*100:.0f}%")
    print(f"  Edad media:      {subset['edad_al_matricularse'].mean():.1f}")

print("\n✓ Tarea 3 (Aprendizaje No Supervisado) completada.")
