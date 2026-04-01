import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# =============================================================================
# DONNÉES ANNUELLES 2025
# =============================================================================
data = {
    'Base opérationnelle': [
        'BO Forez', 'BO Roannais', 'BO Aubenas-Joyeuse', 'BO Ambérieu - Belley',
        'BO Bourg - Montrevel', 'BO Montélimar', 'BO Gleize', 'BO Heyrieux',
        'BO Roussillon', "BO l'Arbresle", 'BO Crest Die', 'BO Valence',
        'BO Privas-Le Cheylard', 'BO Romans', 'BO Décines', 'BO Saint Vallier',
        'BO Saint Étienne', 'BO Givors', 'BO Oullins', 'BO Annonay',
        'BO Valréas', 'BO Rillieux', 'BO Vénissieux', 'BO Firminy - Saint Bonnet',
        'BO Oyonnax'
    ],
    'Taux composite cumulé': [
        3.93, 4.63, 1.68, 3.24, 10.91, 6.05, 7.16, 9.13,
        6.41, 8.90, 2.89, 4.04, 9.40, 14.61, 3.08, 4.81,
        11.47, 12.85, 12.24, 3.04, 11.77, 4.44, 12.98, 8.05, 0.00
    ],
    'Nouveau Score Satisfaction': [
        85.02, 85.31, 90.10, 88.89, 80.49, 82.91, 78.39, 78.61,
        90.00, 80.00, 87.41, 85.52, 87.86, 78.43, 84.96, 85.61,
        77.14, 73.91, 72.14, 88.07, 83.33, 78.95, 76.04, 74.42, 95.00
    ],
    'Poids Global': [
        1.28, 1.56, 0.20, 0.45, 1.71, 0.89, 1.56, 1.56,
        0.27, 0.89, 0.55, 0.66, 0.45, 1.01, 0.55, 0.55,
        1.28, 1.71, 1.71, 0.27, 0.36, 0.55, 0.77, 0.77, 0.00
    ],
    'Nouveau Poids Global': [
        1.74, 1.73, 1.30, 1.25, 1.21, 1.21, 1.12, 1.02,
        0.99, 0.87, 0.80, 0.79, 0.78, 0.75, 0.69, 0.69,
        0.64, 0.59, 0.58, 0.54, 0.41, 0.37, 0.36, 0.29, 0.13
    ],
    'est_PDTS': [
        14, 16, 4, 7, 17, 11, 16, 16,
        5, 11, 8, 9, 7, 12, 8, 8,
        14, 17, 17, 5, 6, 8, 10, 10, 0
    ],
    'Nombre enquêtes envoyées': [
        1609, 1411, 1130, 1353, 1489, 1251, 1324, 1106,
        1229, 1126, 922, 1083, 755, 990, 987, 817,
        1157, 1098, 1064, 844, 571, 873, 785, 566, 285
    ],
    'Taux répondants cumulé': [
        15.35, 17.36, 16.99, 13.97, 13.77, 15.91, 15.03, 16.91,
        13.02, 14.65, 15.51, 13.39, 18.54, 15.45, 13.48, 16.16,
        12.10, 12.57, 13.16, 12.91, 16.81, 10.88, 12.23, 15.19, 14.04
    ]
}

df = pd.DataFrame(data)

# Remplacement des volumes annuels par la moyenne mobile journalière (si CSV disponible)
volume_csv_candidates = [
    Path('/workspace/Nombre_d_enqu_tes_envoy_es_en_moyenne_par_jour_par_BO.csv'),
    Path('/workspace/data/Nombre_d_enqu_tes_envoy_es_en_moyenne_par_jour_par_BO.csv'),
    Path('/home/ubuntu/.cursor/projects/workspace/uploads/Nombre_d_enqu_tes_envoy_es_en_moyenne_par_jour_par_BO.csv'),
]
volume_csv_path = next((p for p in volume_csv_candidates if p.exists()), None)

if volume_csv_path is not None:
    mm_df = pd.read_csv(volume_csv_path)
    daily_col = "Nombre d'enquêtes envoyées MM"
    bo_col = 'Base opérationnelle'

    if bo_col not in mm_df.columns or daily_col not in mm_df.columns:
        raise ValueError(
            f"Colonnes attendues absentes dans {volume_csv_path}: {bo_col}, {daily_col}"
        )

    mean_daily = mm_df.groupby(bo_col)[daily_col].mean()
    annualized = (mean_daily * 365).round().astype(int)

    missing = set(df['Base opérationnelle']) - set(annualized.index)
    if missing:
        raise ValueError(
            "BO manquantes dans le CSV de moyenne mobile: " + ", ".join(sorted(missing))
        )

    df['Nombre enquêtes envoyées MM / jour'] = df['Base opérationnelle'].map(mean_daily).round(2)
    df['Nombre enquêtes envoyées'] = (
        df['Base opérationnelle'].map(annualized).astype(int)
    )
    print(f"Volumes d'enquêtes chargés depuis la moyenne mobile: {volume_csv_path}")
else:
    print("⚠️ CSV de moyenne mobile non trouvé. Volumes annuels historiques conservés.")

# Volumes estimés pour le challenge
df['Nombre répondants estimé annuel'] = (df['Nombre enquêtes envoyées'] * df['Taux répondants cumulé'] / 100).round(0).astype(int)
df['Enquêtes / mois'] = (df['Nombre enquêtes envoyées'] / 12).round(0).astype(int)
df['Enquêtes / 2 mois'] = (df['Nombre enquêtes envoyées'] / 6).round(0).astype(int)
df['Répondants / mois'] = (df['Nombre répondants estimé annuel'] / 12).round(0).astype(int)
df['Répondants / 2 mois'] = (df['Nombre répondants estimé annuel'] / 6).round(0).astype(int)
df['PDTS / mois'] = (df['est_PDTS'] / 12).round(1)
df['PDTS / 2 mois'] = (df['est_PDTS'] / 6).round(1)

# =============================================================================
# CLUSTERING EN 4 GROUPES
# =============================================================================
features_for_clustering = [
    'Nombre enquêtes envoyées',
    'Taux répondants cumulé',
    'est_PDTS',
    'Nouveau Score Satisfaction'
]

X = df[features_for_clustering].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

N_GROUPS = 4

# Comparaison K-Means vs Hiérarchique pour k=4
kmeans_final = KMeans(n_clusters=N_GROUPS, random_state=42, n_init=30)
labels_kmeans = kmeans_final.fit_predict(X_scaled)
sil_km = silhouette_score(X_scaled, labels_kmeans)

agg = AgglomerativeClustering(n_clusters=N_GROUPS, linkage='ward')
labels_agg = agg.fit_predict(X_scaled)
sil_agg = silhouette_score(X_scaled, labels_agg)

print(f"Silhouette K-Means (k=4): {sil_km:.4f}")
print(f"Silhouette Hiérarchique (k=4): {sil_agg:.4f}")

# On prend le meilleur
if sil_agg > sil_km:
    labels_best = labels_agg
    method = "Hiérarchique (Ward)"
else:
    labels_best = labels_kmeans
    method = "K-Means"

print(f"Méthode retenue: {method}")

df['Groupe_raw'] = labels_best

# Relabeler par score de satisfaction décroissant
group_means = df.groupby('Groupe_raw')['Nouveau Score Satisfaction'].mean().sort_values(ascending=False)
label_mapping = {old: new+1 for new, old in enumerate(group_means.index)}
df['Groupe'] = df['Groupe_raw'].map(label_mapping)

# Nommage
group_names = {
    1: "Groupe A",
    2: "Groupe B",
    3: "Groupe C",
    4: "Groupe D"
}
# On affine le nommage après analyse
for g in sorted(df['Groupe'].unique()):
    sub = df[df['Groupe'] == g]
    avg_sat = sub['Nouveau Score Satisfaction'].mean()
    avg_vol = sub['Enquêtes / 2 mois'].mean()
    n = len(sub)
    group_names[g] = f"Groupe {chr(64+g)} ({n} BOs)"

df['Nom Groupe'] = df['Groupe'].map(group_names)

# =============================================================================
# VISUALISATIONS
# =============================================================================
sns.set_theme(style="whitegrid", font_scale=1.1)
colors_4 = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']

# --- Fig 1 : Scatter Score vs Volume mensuel avec groupes ---
fig, ax = plt.subplots(figsize=(16, 10))
for g in sorted(df['Groupe'].unique()):
    sub = df[df['Groupe'] == g]
    color = colors_4[g-1]
    ax.scatter(sub['Enquêtes / 2 mois'], sub['Nouveau Score Satisfaction'],
               s=sub['est_PDTS']*20 + 80, alpha=0.8, color=color,
               edgecolors='white', linewidth=1.5, label=group_names[g], zorder=3)
    for _, row in sub.iterrows():
        name = row['Base opérationnelle'].replace('BO ', '')
        ax.annotate(name, (row['Enquêtes / 2 mois'], row['Nouveau Score Satisfaction']),
                    fontsize=8, ha='center', va='bottom',
                    xytext=(0, 8), textcoords='offset points',
                    fontweight='bold')

ax.set_xlabel('Enquêtes estimées sur 2 mois (période challenge)', fontsize=13, fontweight='bold')
ax.set_ylabel('Nouveau Score de Satisfaction (%)', fontsize=13, fontweight='bold')
ax.set_title('4 Groupes de BOs Comparables pour le Challenge\n(taille des points = nb de PDTS annuel)', fontsize=15, fontweight='bold')
ax.legend(loc='lower left', fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/workspace/01_groupes_bo_scatter.png', dpi=150, bbox_inches='tight')
plt.close()

# --- Fig 2 : Score détaillé par BO dans chaque groupe ---
n_groups = len(df['Groupe'].unique())
fig, axes = plt.subplots(1, n_groups, figsize=(5*n_groups, 7), sharey=True)
if not hasattr(axes, '__len__'):
    axes = [axes]

for idx, g in enumerate(sorted(df['Groupe'].unique())):
    sub = df[df['Groupe'] == g].sort_values('Nouveau Score Satisfaction', ascending=True)
    ax = axes[idx]
    color = colors_4[g-1]

    y_pos = range(len(sub))
    bars = ax.barh(y_pos, sub['Nouveau Score Satisfaction'], color=color, alpha=0.8, edgecolor='white')
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels([bo.replace('BO ', '') for bo in sub['Base opérationnelle']], fontsize=9)
    ax.set_xlabel('Score Satisfaction (%)', fontsize=10)
    ax.set_title(group_names[g], fontsize=11, fontweight='bold', pad=10)
    ax.set_xlim(65, 100)

    for bar, (_, row) in zip(bars, sub.iterrows()):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f"{row['Nouveau Score Satisfaction']:.1f}%", va='center', fontsize=8)

plt.suptitle('Score de Satisfaction Actuel par Groupe', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('/workspace/02_scores_par_groupe.png', dpi=150, bbox_inches='tight')
plt.close()

# --- Fig 3 : Heatmap profil moyen ---
metrics = ['Nouveau Score Satisfaction', 'Taux composite cumulé', 'Taux répondants cumulé',
           'est_PDTS', 'Enquêtes / 2 mois', 'Répondants / 2 mois', 'PDTS / 2 mois']
group_summary = df.groupby('Groupe')[metrics].mean()
group_summary.index = [group_names[g] for g in group_summary.index]

fig, ax = plt.subplots(figsize=(16, 5))
normalized = group_summary.copy()
for col in normalized.columns:
    col_min, col_max = normalized[col].min(), normalized[col].max()
    if col_max > col_min:
        normalized[col] = (normalized[col] - col_min) / (col_max - col_min)
    else:
        normalized[col] = 0.5

display_labels = ['Score Satisfaction\n(%)', 'Taux Composite\n(ancien, %)', 'Taux Répondants\n(%)',
                  'PDTS\n(annuel)', 'Enquêtes\n(/2 mois)', 'Répondants\n(/2 mois)', 'PDTS\n(/2 mois)']

sns.heatmap(normalized, annot=group_summary.round(1).values, fmt='', cmap='YlOrRd_r',
            ax=ax, linewidths=2, linecolor='white',
            xticklabels=display_labels,
            cbar_kws={'label': 'Performance relative'})
ax.set_title('Profil Moyen par Groupe — Volumes adaptés à une période de 2 mois', fontsize=14, fontweight='bold')
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
plt.tight_layout()
plt.savefig('/workspace/03_heatmap_groupes.png', dpi=150, bbox_inches='tight')
plt.close()

# --- Fig 4 : Analyse multi-vues ---
fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# 4a: Score vs Taux Répondants
ax1 = axes[0, 0]
for g in sorted(df['Groupe'].unique()):
    sub = df[df['Groupe'] == g]
    ax1.scatter(sub['Taux répondants cumulé'], sub['Nouveau Score Satisfaction'],
                s=120, color=colors_4[g-1], alpha=0.8, label=group_names[g],
                edgecolors='white', linewidth=1)
    for _, row in sub.iterrows():
        ax1.annotate(row['Base opérationnelle'].replace('BO ', ''),
                     (row['Taux répondants cumulé'], row['Nouveau Score Satisfaction']),
                     fontsize=7, ha='center', va='bottom', xytext=(0, 5), textcoords='offset points')
ax1.set_xlabel('Taux de Répondants (%)')
ax1.set_ylabel('Score Satisfaction (%)')
ax1.set_title('Score Satisfaction vs Taux Répondants')
ax1.legend(fontsize=8)

# 4b: Score vs PDTS
ax2 = axes[0, 1]
for g in sorted(df['Groupe'].unique()):
    sub = df[df['Groupe'] == g]
    ax2.scatter(sub['est_PDTS'], sub['Nouveau Score Satisfaction'],
                s=120, color=colors_4[g-1], alpha=0.8, label=group_names[g],
                edgecolors='white', linewidth=1)
    for _, row in sub.iterrows():
        ax2.annotate(row['Base opérationnelle'].replace('BO ', ''),
                     (row['est_PDTS'], row['Nouveau Score Satisfaction']),
                     fontsize=7, ha='center', va='bottom', xytext=(0, 5), textcoords='offset points')
ax2.set_xlabel('Nombre de PDTS (réponses 1/4) — annuel')
ax2.set_ylabel('Score Satisfaction (%)')
ax2.set_title('Score Satisfaction vs Nombre de PDTS')
ax2.legend(fontsize=8)

# 4c: Volumes mensuels par BO
ax3 = axes[1, 0]
sorted_df = df.sort_values('Groupe')
colors_bars = [colors_4[g-1] for g in sorted_df['Groupe']]
bars = ax3.bar(range(len(sorted_df)), sorted_df['Enquêtes / 2 mois'], color=colors_bars, alpha=0.8, edgecolor='white')
ax3.set_xticks(range(len(sorted_df)))
ax3.set_xticklabels([bo.replace('BO ', '') for bo in sorted_df['Base opérationnelle']], rotation=90, fontsize=7)
ax3.set_ylabel('Enquêtes sur 2 mois (estimé)')
ax3.set_title('Volume d\'enquêtes estimé sur la période challenge (2 mois)')
handles = [mpatches.Patch(color=colors_4[g-1], label=group_names[g]) for g in sorted(df['Groupe'].unique())]
ax3.legend(handles=handles, fontsize=7, loc='upper right')

# 4d: Boxplot des scores par groupe
ax4 = axes[1, 1]
group_data = [df[df['Groupe'] == g]['Nouveau Score Satisfaction'].values for g in sorted(df['Groupe'].unique())]
bp = ax4.boxplot(group_data, patch_artist=True, tick_labels=[f'Groupe {chr(64+g)}' for g in sorted(df['Groupe'].unique())])
for patch, g in zip(bp['boxes'], sorted(df['Groupe'].unique())):
    patch.set_facecolor(colors_4[g-1])
    patch.set_alpha(0.7)
ax4.set_ylabel('Score Satisfaction (%)')
ax4.set_title('Distribution du Score Satisfaction par Groupe')

plt.suptitle('Analyse Détaillée — Challenge Satisfaction Client', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('/workspace/04_analyse_detaillee.png', dpi=150, bbox_inches='tight')
plt.close()

# --- Fig 5 : Système de scoring du challenge ---
fig, ax = plt.subplots(figsize=(14, 8))
ax.axis('off')

scoring_text = """
SYSTÈME DE SCORE DU CHALLENGE — 100 points max par BO

═══════════════════════════════════════════════════════════════════════════

  1. PROGRESSION DU SCORE SATISFACTION                         40 points
     ─────────────────────────────────────────────────────
     Δ Score = Score fin challenge − Score début challenge
     Points = (Δ Score / Meilleur Δ du groupe) × 40
     → Récompense la BO qui progresse le plus dans son groupe

  2. RÉDUCTION DES PDTS                                        30 points
     ─────────────────────────────────────────────────────
     Δ PDTS = Taux PDTS début − Taux PDTS fin  (en % des répondants)
     Points = (Δ PDTS / Meilleur Δ du groupe) × 30
     → Utilise un TAUX (pas un nombre absolu) pour être équitable
       entre BOs de volumes différents

  3. PROGRESSION DU TAUX DE RÉPONDANTS                         20 points
     ─────────────────────────────────────────────────────
     Δ Taux = Taux répondants fin − Taux répondants début
     Points = (Δ Taux / Meilleur Δ du groupe) × 20
     → Encourage l'engagement client

  4. BONUS "EXCELLENCE"                                        10 points
     ─────────────────────────────────────────────────────
     +5 pts si Score Satisfaction fin ≥ 90%
     +5 pts si Taux Répondants fin ≥ 18%
     → Récompense aussi le niveau atteint, pas que la progression

═══════════════════════════════════════════════════════════════════════════
                                               TOTAL MAX : 100 points
  Le GAGNANT par groupe = la BO avec le score total le plus élevé
"""

ax.text(0.05, 0.95, scoring_text, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='#f0f4ff', alpha=0.9, edgecolor='#2196F3', linewidth=2))

ax.set_title('Système de Score du Challenge', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('/workspace/05_systeme_scoring.png', dpi=150, bbox_inches='tight')
plt.close()

# =============================================================================
# AFFICHAGE CONSOLE
# =============================================================================
print("\n" + "="*80)
print("GROUPES DE BASES OPÉRATIONNELLES COMPARABLES — 4 GROUPES")
print("="*80)

for g in sorted(df['Groupe'].unique()):
    sub = df[df['Groupe'] == g].sort_values('Nouveau Score Satisfaction', ascending=False)
    print(f"\n{'─'*75}")
    print(f"  {group_names[g]}")
    print(f"{'─'*75}")
    print(f"  Moy. Score Satisfaction : {sub['Nouveau Score Satisfaction'].mean():.2f}%")
    print(f"  Moy. Taux Répondants   : {sub['Taux répondants cumulé'].mean():.2f}%")
    print(f"  Moy. PDTS (annuel)     : {sub['est_PDTS'].mean():.1f}  →  /2 mois: {sub['PDTS / 2 mois'].mean():.1f}")
    print(f"  Moy. Enquêtes /2 mois  : {sub['Enquêtes / 2 mois'].mean():.0f}")
    print(f"  Moy. Répondants /2 mois: {sub['Répondants / 2 mois'].mean():.0f}")
    print()
    header = f"    {'BO':<28} {'Score':>7} {'PDTS':>5} {'Enq/2m':>7} {'Rép/2m':>7} {'Tx Rép':>7}"
    print(header)
    print(f"    {'─'*65}")
    for _, row in sub.iterrows():
        name = row['Base opérationnelle'].replace('BO ', '')
        print(f"    {name:<28} {row['Nouveau Score Satisfaction']:>6.1f}% {row['est_PDTS']:>4}  "
              f"{row['Enquêtes / 2 mois']:>6}  {row['Répondants / 2 mois']:>6}  "
              f"{row['Taux répondants cumulé']:>6.1f}%")

print("\n\n" + "="*80)
print("SYSTÈME DE SCORE DU CHALLENGE (100 pts max)")
print("="*80)
print("""
  1. Progression du Score Satisfaction ................ 40 pts
     (Δ Score / Meilleur Δ du groupe) × 40

  2. Réduction du Taux de PDTS ....................... 30 pts
     (Δ Taux PDTS / Meilleur Δ du groupe) × 30
     Attention : on utilise un TAUX, pas un nb absolu

  3. Progression du Taux de Répondants ............... 20 pts
     (Δ Taux / Meilleur Δ du groupe) × 20

  4. Bonus "Excellence" .............................. 10 pts
     +5 si Score fin ≥ 90%
     +5 si Taux Répondants fin ≥ 18%

  → 1 gagnant par groupe = score total le plus élevé
""")

# Export CSV
output_cols = ['Base opérationnelle', 'Groupe', 'Nom Groupe',
               'Nouveau Score Satisfaction', 'Taux composite cumulé',
               'est_PDTS', 'PDTS / 2 mois',
               'Nombre enquêtes envoyées', 'Enquêtes / 2 mois',
               'Taux répondants cumulé', 'Répondants / 2 mois',
               'Nouveau Poids Global']
output = df[output_cols].sort_values(['Groupe', 'Nouveau Score Satisfaction'], ascending=[True, False])
output.to_csv('/workspace/groupes_bo_comparables.csv', index=False, sep=';')
print(f"Fichier CSV exporté: groupes_bo_comparables.csv")
print(f"Graphiques exportés: 01 à 05")
