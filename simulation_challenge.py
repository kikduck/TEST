import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

np.random.seed(42)

# =============================================================================
# DONNÉES DE BASE (annuelles 2025)
# =============================================================================
data = {
    'BO': [
        'BO Forez', 'BO Roannais', 'BO Aubenas-Joyeuse', 'BO Ambérieu - Belley',
        'BO Bourg - Montrevel', 'BO Montélimar', 'BO Gleize', 'BO Heyrieux',
        'BO Roussillon', "BO l'Arbresle", 'BO Crest Die', 'BO Valence',
        'BO Privas-Le Cheylard', 'BO Romans', 'BO Décines', 'BO Saint Vallier',
        'BO Saint Étienne', 'BO Givors', 'BO Oullins', 'BO Annonay',
        'BO Valréas', 'BO Rillieux', 'BO Vénissieux', 'BO Firminy - Saint Bonnet',
        'BO Oyonnax'
    ],
    'Score_ini': [
        85.02, 85.31, 90.10, 88.89, 80.49, 82.91, 78.39, 78.61,
        90.00, 80.00, 87.41, 85.52, 87.86, 78.43, 84.96, 85.61,
        77.14, 73.91, 72.14, 88.07, 83.33, 78.95, 76.04, 74.42, 95.00
    ],
    'est_PDTS_annuel': [
        14, 16, 4, 7, 17, 11, 16, 16,
        5, 11, 8, 9, 7, 12, 8, 8,
        14, 17, 17, 5, 6, 8, 10, 10, 0
    ],
    'Enquetes_annuel': [
        1609, 1411, 1130, 1353, 1489, 1251, 1324, 1106,
        1229, 1126, 922, 1083, 755, 990, 987, 817,
        1157, 1098, 1064, 844, 571, 873, 785, 566, 285
    ],
    'Tx_rep_ini': [
        15.35, 17.36, 16.99, 13.97, 13.77, 15.91, 15.03, 16.91,
        13.02, 14.65, 15.51, 13.39, 18.54, 15.45, 13.48, 16.16,
        12.10, 12.57, 13.16, 12.91, 16.81, 10.88, 12.23, 15.19, 14.04
    ],
    'Groupe': [
        3, 3, 1, 2, 3, 3, 3, 3,
        2, 3, 1, 2, 1, 3, 2, 1,
        4, 4, 4, 2, 1, 4, 4, 4, 1
    ]
}

df = pd.DataFrame(data)

group_labels = {1: "Groupe A", 2: "Groupe B", 3: "Groupe C", 4: "Groupe D"}
colors_4 = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']

# Priorité aux groupes/volumes recalculés (export de analyse_groupes_bo.py)
groupes_csv_path = Path('/workspace/groupes_bo_comparables.csv')
if groupes_csv_path.exists():
    groupes_df = pd.read_csv(groupes_csv_path, sep=';').rename(
        columns={'Base opérationnelle': 'BO'}
    )
    required_cols = {
        'BO',
        'Groupe',
        'Nouveau Score Satisfaction',
        'est_PDTS',
        'Nombre enquêtes envoyées',
        'Taux répondants cumulé',
    }
    missing_cols = required_cols - set(groupes_df.columns)
    if missing_cols:
        raise ValueError(
            f"Colonnes manquantes dans {groupes_csv_path}: {', '.join(sorted(missing_cols))}"
        )

    groups_indexed = groupes_df.set_index('BO')
    missing_bos = set(df['BO']) - set(groups_indexed.index)
    if missing_bos:
        raise ValueError(
            "BO absentes de groupes_bo_comparables.csv: " + ", ".join(sorted(missing_bos))
        )

    df['Score_ini'] = df['BO'].map(groups_indexed['Nouveau Score Satisfaction']).astype(float)
    df['est_PDTS_annuel'] = df['BO'].map(groups_indexed['est_PDTS']).astype(int)
    df['Enquetes_annuel'] = df['BO'].map(groups_indexed['Nombre enquêtes envoyées']).astype(int)
    df['Tx_rep_ini'] = df['BO'].map(groups_indexed['Taux répondants cumulé']).astype(float)
    df['Groupe'] = df['BO'].map(groups_indexed['Groupe']).astype(int)
    print(f"Données simulation synchronisées depuis: {groupes_csv_path}")
else:
    print("⚠️ groupes_bo_comparables.csv introuvable: utilisation des données statiques.")

# =============================================================================
# SYSTÈME DE POINTS PAR RÉPONSE
# =============================================================================
PTS_4 = 6    # 4/4 (très satisfait)
PTS_3 = 3    # 3/4 (satisfait)
PTS_2 = -1   # 2/4 (peu satisfait)
PTS_1 = -15  # 1/4 (insatisfait / PDTS)

# CLASSEMENT = Score moyen par répondant (total pts / nb répondants)
# PRIX EFFORT = meilleure progression du score moyen, tous groupes confondus

print("=" * 90)
print("RÈGLES DU CHALLENGE")
print("=" * 90)
print(f"\n  BARÈME PAR RÉPONSE :")
print(f"    4/4 (très satisfait) : +{PTS_4} pts")
print(f"    3/4 (satisfait)      : +{PTS_3} pts")
print(f"    2/4 (peu satisfait)  : {PTS_2} pt")
print(f"    1/4 (insatisfait)    : {PTS_1} pts")
print(f"\n  CLASSEMENT = Score moyen par répondant = Total pts ÷ Nb répondants")
print(f"  → 1 gagnant par groupe = meilleur score moyen /répondant en fin de challenge")
print(f"\n  PRIX EFFORT = 1 prix unique tous groupes confondus")
print(f"  → La BO avec la plus forte progression de score moyen /répondant")
print(f"  → Récompense l'effort, pas le niveau de départ")

# =============================================================================
# RECONSTRUCTION DES NOTES 4/4, 3/4, 2/4, 1/4 PAR BO
# =============================================================================
df['Rep_annuel'] = (df['Enquetes_annuel'] * df['Tx_rep_ini'] / 100).round(0).astype(int)
df['Enq_2m'] = (df['Enquetes_annuel'] / 6).round(0).astype(int)
df['Rep_2m'] = (df['Rep_annuel'] / 6).round(0).astype(int)
df['PDTS_2m'] = (df['est_PDTS_annuel'] / 6).round(0).astype(int)

df['Tx_1'] = np.where(df['Rep_2m'] > 0, df['PDTS_2m'] / df['Rep_2m'], 0)
df['Tx_34_sum'] = df['Score_ini'] / 100 + df['Tx_1']

df['ratio_4_in_43'] = 0.40 + 0.25 * (df['Score_ini'] - 70) / 30
df['ratio_4_in_43'] = df['ratio_4_in_43'].clip(0.35, 0.65)

df['Tx_4'] = df['Tx_34_sum'] * df['ratio_4_in_43']
df['Tx_3'] = df['Tx_34_sum'] * (1 - df['ratio_4_in_43'])
df['Tx_2'] = 1 - df['Tx_4'] - df['Tx_3'] - df['Tx_1']
df['Tx_2'] = df['Tx_2'].clip(lower=0.01)

df['N4_ini'] = (df['Rep_2m'] * df['Tx_4']).round(0).astype(int)
df['N3_ini'] = (df['Rep_2m'] * df['Tx_3']).round(0).astype(int)
df['N1_ini'] = df['PDTS_2m']
df['N2_ini'] = df['Rep_2m'] - df['N4_ini'] - df['N3_ini'] - df['N1_ini']
df['N2_ini'] = df['N2_ini'].clip(lower=0)

df['Rep_2m_ini'] = df['N4_ini'] + df['N3_ini'] + df['N2_ini'] + df['N1_ini']
df['Pts_ini'] = df['N4_ini'] * PTS_4 + df['N3_ini'] * PTS_3 + df['N2_ini'] * PTS_2 + df['N1_ini'] * PTS_1
df['Score_pts_ini'] = np.where(df['Rep_2m_ini'] > 0, df['Pts_ini'] / df['Rep_2m_ini'], 0)

# =============================================================================
# SIMULATION "APRÈS CHALLENGE" (2 mois)
# =============================================================================
sim_params = {
    'BO Oyonnax':              {'delta_tx_rep': +3.5, 'pdts_fin': 0, 'convert_2_to_3': 1, 'new_rep_pct_4': 0.55, 'new_rep_pct_3': 0.35},
    'BO Aubenas-Joyeuse':      {'delta_tx_rep': +2.0, 'pdts_fin': 0, 'convert_2_to_3': 2, 'new_rep_pct_4': 0.60, 'new_rep_pct_3': 0.30},
    'BO Privas-Le Cheylard':   {'delta_tx_rep': +1.0, 'pdts_fin': 1, 'convert_2_to_3': 1, 'new_rep_pct_4': 0.50, 'new_rep_pct_3': 0.35},
    'BO Crest Die':            {'delta_tx_rep': +2.5, 'pdts_fin': 1, 'convert_2_to_3': 2, 'new_rep_pct_4': 0.55, 'new_rep_pct_3': 0.30},
    'BO Saint Vallier':        {'delta_tx_rep': +1.5, 'pdts_fin': 0, 'convert_2_to_3': 1, 'new_rep_pct_4': 0.55, 'new_rep_pct_3': 0.35},
    'BO Valréas':              {'delta_tx_rep': +3.0, 'pdts_fin': 0, 'convert_2_to_3': 1, 'new_rep_pct_4': 0.60, 'new_rep_pct_3': 0.30},

    'BO Roussillon':           {'delta_tx_rep': +2.0, 'pdts_fin': 0, 'convert_2_to_3': 1, 'new_rep_pct_4': 0.55, 'new_rep_pct_3': 0.35},
    'BO Ambérieu - Belley':    {'delta_tx_rep': +3.0, 'pdts_fin': 1, 'convert_2_to_3': 2, 'new_rep_pct_4': 0.55, 'new_rep_pct_3': 0.30},
    'BO Annonay':              {'delta_tx_rep': +2.5, 'pdts_fin': 0, 'convert_2_to_3': 1, 'new_rep_pct_4': 0.60, 'new_rep_pct_3': 0.30},
    'BO Valence':              {'delta_tx_rep': +1.5, 'pdts_fin': 1, 'convert_2_to_3': 1, 'new_rep_pct_4': 0.50, 'new_rep_pct_3': 0.35},
    'BO Décines':              {'delta_tx_rep': +3.5, 'pdts_fin': 1, 'convert_2_to_3': 2, 'new_rep_pct_4': 0.55, 'new_rep_pct_3': 0.30},

    'BO Roannais':             {'delta_tx_rep': +1.5, 'pdts_fin': 2, 'convert_2_to_3': 3, 'new_rep_pct_4': 0.50, 'new_rep_pct_3': 0.35},
    'BO Forez':                {'delta_tx_rep': +2.0, 'pdts_fin': 1, 'convert_2_to_3': 4, 'new_rep_pct_4': 0.55, 'new_rep_pct_3': 0.30},
    'BO Montélimar':           {'delta_tx_rep': +1.0, 'pdts_fin': 1, 'convert_2_to_3': 2, 'new_rep_pct_4': 0.50, 'new_rep_pct_3': 0.35},
    'BO Bourg - Montrevel':    {'delta_tx_rep': +2.5, 'pdts_fin': 2, 'convert_2_to_3': 3, 'new_rep_pct_4': 0.50, 'new_rep_pct_3': 0.30},
    "BO l'Arbresle":           {'delta_tx_rep': +1.0, 'pdts_fin': 1, 'convert_2_to_3': 2, 'new_rep_pct_4': 0.45, 'new_rep_pct_3': 0.35},
    'BO Heyrieux':             {'delta_tx_rep': +2.0, 'pdts_fin': 1, 'convert_2_to_3': 3, 'new_rep_pct_4': 0.55, 'new_rep_pct_3': 0.30},
    'BO Romans':               {'delta_tx_rep': +3.0, 'pdts_fin': 1, 'convert_2_to_3': 3, 'new_rep_pct_4': 0.55, 'new_rep_pct_3': 0.30},
    'BO Gleize':               {'delta_tx_rep': +1.5, 'pdts_fin': 2, 'convert_2_to_3': 2, 'new_rep_pct_4': 0.45, 'new_rep_pct_3': 0.35},

    'BO Rillieux':             {'delta_tx_rep': +4.0, 'pdts_fin': 1, 'convert_2_to_3': 2, 'new_rep_pct_4': 0.50, 'new_rep_pct_3': 0.30},
    'BO Saint Étienne':        {'delta_tx_rep': +2.0, 'pdts_fin': 2, 'convert_2_to_3': 2, 'new_rep_pct_4': 0.45, 'new_rep_pct_3': 0.35},
    'BO Vénissieux':           {'delta_tx_rep': +3.0, 'pdts_fin': 1, 'convert_2_to_3': 2, 'new_rep_pct_4': 0.50, 'new_rep_pct_3': 0.30},
    'BO Firminy - Saint Bonnet': {'delta_tx_rep': +2.5, 'pdts_fin': 1, 'convert_2_to_3': 1, 'new_rep_pct_4': 0.50, 'new_rep_pct_3': 0.35},
    'BO Givors':               {'delta_tx_rep': +1.5, 'pdts_fin': 2, 'convert_2_to_3': 2, 'new_rep_pct_4': 0.45, 'new_rep_pct_3': 0.35},
    'BO Oullins':              {'delta_tx_rep': +2.0, 'pdts_fin': 2, 'convert_2_to_3': 2, 'new_rep_pct_4': 0.45, 'new_rep_pct_3': 0.30},
}

results = []
for _, row in df.iterrows():
    bo = row['BO']
    s = sim_params[bo]
    enq_2m = row['Enq_2m']

    n4_i, n3_i, n2_i, n1_i = row['N4_ini'], row['N3_ini'], row['N2_ini'], row['N1_ini']
    rep_i = row['Rep_2m_ini']
    pts_i = row['Pts_ini']
    score_pts_i = row['Score_pts_ini']

    tx_rep_fin = row['Tx_rep_ini'] + s['delta_tx_rep']
    rep_fin_total = int(round(enq_2m * tx_rep_fin / 100))
    new_rep = max(0, rep_fin_total - rep_i)

    n1_f = s['pdts_fin']
    pdts_reduits = max(0, n1_i - n1_f)
    convert = min(s['convert_2_to_3'], n2_i)

    n4_from_pdts = int(round(pdts_reduits * 0.4))
    n3_from_pdts = int(round(pdts_reduits * 0.4))
    n2_from_pdts = pdts_reduits - n4_from_pdts - n3_from_pdts

    n4_base = n4_i + n4_from_pdts
    n3_base = n3_i + convert + n3_from_pdts
    n2_base = max(0, n2_i - convert + n2_from_pdts)

    n4_new = int(round(new_rep * s['new_rep_pct_4']))
    n3_new = int(round(new_rep * s['new_rep_pct_3']))
    n2_new = new_rep - n4_new - n3_new

    n4_f = n4_base + n4_new
    n3_f = n3_base + n3_new
    n2_f = max(0, n2_base + n2_new)
    rep_f = n4_f + n3_f + n2_f + n1_f

    pts_f = n4_f * PTS_4 + n3_f * PTS_3 + n2_f * PTS_2 + n1_f * PTS_1
    score_pts_f = pts_f / rep_f if rep_f > 0 else 0

    results.append({
        'BO': bo, 'Groupe': row['Groupe'], 'Enq_2m': enq_2m,
        'N4_ini': n4_i, 'N3_ini': n3_i, 'N2_ini': n2_i, 'N1_ini': n1_i,
        'Rep_ini': rep_i, 'Pts_ini': pts_i, 'Score_pts_ini': round(score_pts_i, 2),
        'Tx_rep_ini': round(row['Tx_rep_ini'], 2),
        'N4_fin': n4_f, 'N3_fin': n3_f, 'N2_fin': n2_f, 'N1_fin': n1_f,
        'Rep_fin': rep_f, 'Pts_fin': pts_f, 'Score_pts_fin': round(score_pts_f, 2),
        'Tx_rep_fin': round(tx_rep_fin, 2),
        'Delta_score_pts': round(score_pts_f - score_pts_i, 2),
    })

sim_df = pd.DataFrame(results)

# =============================================================================
# CLASSEMENT SIMPLIFIÉ
# =============================================================================
# Gagnant par groupe = meilleur score moyen /répondant en fin de challenge
sim_df['Rang'] = sim_df.groupby('Groupe')['Score_pts_fin'].rank(ascending=False, method='min').astype(int)
sim_df['Gagnant'] = sim_df['Rang'] == 1

# Prix Effort = meilleure progression tous groupes confondus
best_effort_idx = sim_df['Delta_score_pts'].idxmax()
sim_df['Prix_Effort'] = False
sim_df.loc[best_effort_idx, 'Prix_Effort'] = True

# Top 3 effort pour le tableau
top3_effort = sim_df.nlargest(3, 'Delta_score_pts')

# =============================================================================
# AFFICHAGE CONSOLE
# =============================================================================
print("\n\n" + "=" * 100)
print("SIMULATION CHALLENGE — RÉSULTATS AVANT / APRÈS (période 2 mois)")
print("=" * 100)

for g in sorted(sim_df['Groupe'].unique()):
    grp = sim_df[sim_df['Groupe'] == g].sort_values('Score_pts_fin', ascending=False)
    winner = grp.iloc[0]['BO']

    print(f"\n{'━' * 100}")
    print(f"  {group_labels[g]} ({len(grp)} BOs) — GAGNANT : {winner}")
    print(f"{'━' * 100}")

    print(f"\n  {'BO':<28} │ {'── AVANT ──':^24} │ {'── APRÈS ──':^24} │ {'RÉSULTAT':^20}")
    print(f"  {'':<28} │ {'4/4':>4} {'3/4':>4} {'2/4':>4} {'1/4':>4} {'Rép':>4} {'Moy':>6} │ {'4/4':>4} {'3/4':>4} {'2/4':>4} {'1/4':>4} {'Rép':>4} {'Moy':>6} │ {'Δ Moy':>6} {'Rang':>5} ")
    print(f"  {'─' * 28}─┼─{'─' * 24}─┼─{'─' * 24}─┼─{'─' * 20}")

    for _, r in grp.iterrows():
        name = r['BO'].replace('BO ', '')
        flag = " ★" if r['Gagnant'] else ""
        effort = " 🏅" if r['Prix_Effort'] else ""
        print(f"  {name:<28} │ "
              f"{r['N4_ini']:>4} {r['N3_ini']:>4} {r['N2_ini']:>4} {r['N1_ini']:>4} {r['Rep_ini']:>4} {r['Score_pts_ini']:>+6.2f} │ "
              f"{r['N4_fin']:>4} {r['N3_fin']:>4} {r['N2_fin']:>4} {r['N1_fin']:>4} {r['Rep_fin']:>4} {r['Score_pts_fin']:>+6.2f} │ "
              f"{r['Delta_score_pts']:>+5.2f}  {r['Rang']:>3}e{flag}{effort}")

    print()
    print(f"  Moy. score /rép : {grp['Score_pts_ini'].mean():>+.2f} → {grp['Score_pts_fin'].mean():>+.2f}")
    print(f"  PDTS : {grp['N1_ini'].sum()} → {grp['N1_fin'].sum()}")

# Podium et Prix Effort
print("\n\n" + "=" * 100)
print("PODIUM — 4 GAGNANTS + 1 PRIX EFFORT")
print("=" * 100)

for g in sorted(sim_df['Groupe'].unique()):
    w = sim_df[(sim_df['Groupe'] == g) & (sim_df['Gagnant'])].iloc[0]
    print(f"  ★ {group_labels[g]:>10} :  {w['BO']:<30}  Score: {w['Score_pts_fin']:+.2f} /rép  "
          f"(4/4: {w['N4_ini']}→{w['N4_fin']}  1/4: {w['N1_ini']}→{w['N1_fin']})")

effort_winner = sim_df[sim_df['Prix_Effort']].iloc[0]
print(f"\n  🏅 PRIX EFFORT :  {effort_winner['BO']:<30}  Progression: {effort_winner['Delta_score_pts']:+.2f} /rép  "
      f"({group_labels[effort_winner['Groupe']]})")
print(f"\n  Top 3 progression :")
for i, (_, r) in enumerate(top3_effort.iterrows()):
    print(f"    {i+1}. {r['BO']:<30}  Δ {r['Delta_score_pts']:+.2f} /rép  ({group_labels[r['Groupe']]})")

# =============================================================================
# VISUALISATIONS
# =============================================================================
sns.set_theme(style="whitegrid", font_scale=1.0)

# --- Fig 6 : Score moyen /répondant AVANT / APRÈS ---
fig, ax = plt.subplots(figsize=(20, 13))

sim_sorted = sim_df.sort_values(['Groupe', 'Score_pts_fin'], ascending=[True, False])
bar_height = 0.35

for i, (_, r) in enumerate(sim_sorted.iterrows()):
    color = colors_4[r['Groupe']-1]

    ax.barh(i + bar_height/2, r['Score_pts_ini'], height=bar_height,
            color=color, alpha=0.30, edgecolor=color, linewidth=0.8)
    ax.barh(i - bar_height/2, r['Score_pts_fin'], height=bar_height,
            color=color, alpha=0.85, edgecolor='white', linewidth=0.8)

    ax.text(max(r['Score_pts_fin'], r['Score_pts_ini']) + 0.12, i,
            f"{r['Score_pts_ini']:+.2f} → {r['Score_pts_fin']:+.2f}",
            va='center', fontsize=7.5, fontweight='bold',
            color='#2E7D32' if r['Delta_score_pts'] > 0 else '#C62828')

    if r['Gagnant']:
        ax.text(0.05, i, "★", va='center', fontsize=14, fontweight='bold', color='#FFD600',
                bbox=dict(boxstyle='round,pad=0.15', facecolor=color, alpha=0.9))
    if r['Prix_Effort']:
        ax.text(0.40, i, "PRIX EFFORT", va='center', fontsize=7, fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='#FF6F00', alpha=0.9))

current_g = None
for i, (_, r) in enumerate(sim_sorted.iterrows()):
    if current_g is not None and r['Groupe'] != current_g:
        ax.axhline(y=i - 0.5, color='gray', linewidth=1.5, linestyle='--', alpha=0.5)
    current_g = r['Groupe']

bo_labels = [r['BO'].replace('BO ', '') for _, r in sim_sorted.iterrows()]
ax.set_yticks(range(len(sim_sorted)))
ax.set_yticklabels(bo_labels, fontsize=9)
ax.set_xlabel('Score moyen par répondant (points)', fontsize=12, fontweight='bold')
ax.set_title(f'Classement Challenge — Score Moyen /Répondant\n'
             f'Barème : 4/4 = +{PTS_4}pts | 3/4 = +{PTS_3}pts | 2/4 = {PTS_2}pt | 1/4 = {PTS_1}pts'
             f'   |   ★ = Gagnant du groupe   |   EFFORT = Prix Effort',
             fontsize=12, fontweight='bold')
ax.invert_yaxis()

handles = [mpatches.Patch(color=colors_4[g-1], alpha=0.85, label=group_labels[g]) for g in sorted(sim_df['Groupe'].unique())]
handles.append(mpatches.Patch(color='gray', alpha=0.35, label="Avant"))
handles.append(mpatches.Patch(color='gray', alpha=0.85, label="Après"))
ax.legend(handles=handles, loc='lower right', fontsize=9)
plt.tight_layout()
plt.savefig('/workspace/06_simulation_avant_apres.png', dpi=150, bbox_inches='tight')
plt.close()

# --- Fig 7 : Décomposition des notes AVANT / APRÈS par groupe ---
fig, axes = plt.subplots(2, 2, figsize=(22, 18))

for idx, g in enumerate(sorted(sim_df['Groupe'].unique())):
    ax = axes[idx // 2][idx % 2]
    grp = sim_df[sim_df['Groupe'] == g].sort_values('Score_pts_fin', ascending=True)
    n = len(grp)
    bar_h = 0.35
    y_avant = [i + bar_h/2 + 0.05 for i in range(n)]
    y_apres = [i - bar_h/2 - 0.05 for i in range(n)]

    note_colors = {'4/4': '#1B5E20', '3/4': '#66BB6A', '2/4': '#FFA726', '1/4': '#C62828'}

    for yi, (_, r) in zip(range(n), grp.iterrows()):
        left = 0
        for val, col in [(r['N4_ini'], note_colors['4/4']), (r['N3_ini'], note_colors['3/4']),
                         (r['N2_ini'], note_colors['2/4']), (r['N1_ini'], note_colors['1/4'])]:
            ax.barh(y_avant[yi], val, left=left, height=bar_h, color=col, alpha=0.35, edgecolor='white', linewidth=0.5)
            if val > 1:
                ax.text(left + val/2, y_avant[yi], str(int(val)), ha='center', va='center', fontsize=7, color='gray')
            left += val

        left = 0
        for val, col in [(r['N4_fin'], note_colors['4/4']), (r['N3_fin'], note_colors['3/4']),
                         (r['N2_fin'], note_colors['2/4']), (r['N1_fin'], note_colors['1/4'])]:
            ax.barh(y_apres[yi], val, left=left, height=bar_h, color=col, alpha=0.85, edgecolor='white', linewidth=0.5)
            if val > 1:
                ax.text(left + val/2, y_apres[yi], str(int(val)), ha='center', va='center', fontsize=7, fontweight='bold', color='white')
            left += val

        suffix = ""
        if r['Gagnant']:
            suffix = " [1er]"
        if r['Prix_Effort']:
            suffix += " [EFFORT]"
        ax.text(left + 1, yi, f"{r['Score_pts_ini']:+.2f} → {r['Score_pts_fin']:+.2f}{suffix}",
                va='center', fontsize=8, fontweight='bold')

    bo_short = [bo.replace('BO ', '') for bo in grp['BO']]
    ax.set_yticks(range(n))
    ax.set_yticklabels(bo_short, fontsize=9)
    ax.set_xlabel('Nombre de répondants par note', fontsize=10)

    winner = grp[grp['Gagnant']].iloc[0]['BO'].replace('BO ', '') if grp['Gagnant'].any() else ''
    ax.set_title(f"{group_labels[g]} — ★ Gagnant : {winner}", fontsize=12, fontweight='bold', color=colors_4[g-1])

    handles_notes = [mpatches.Patch(color=note_colors[n], alpha=0.85, label=n) for n in ['4/4', '3/4', '2/4', '1/4']]
    handles_notes.append(mpatches.Patch(color='gray', alpha=0.35, label='Avant'))
    handles_notes.append(mpatches.Patch(color='gray', alpha=0.85, label='Après'))
    ax.legend(handles=handles_notes, fontsize=7, loc='lower right', ncol=3)

plt.suptitle(f'Détail des Notes AVANT / APRÈS par BO — Score = Total pts ÷ Nb répondants\n'
             f'4/4 = +{PTS_4}pts | 3/4 = +{PTS_3}pts | 2/4 = {PTS_2}pt | 1/4 = {PTS_1}pts',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('/workspace/07_decomposition_scoring.png', dpi=150, bbox_inches='tight')
plt.close()

# --- Fig 8 : Classement final + Prix Effort ---
fig, axes = plt.subplots(1, 2, figsize=(20, 10))

# 8a : Classement par score final
ax = axes[0]
sim_ranked = sim_df.sort_values('Score_pts_fin', ascending=True)
colors_c = [colors_4[r['Groupe']-1] for _, r in sim_ranked.iterrows()]
bo_labels = [r['BO'].replace('BO ', '') for _, r in sim_ranked.iterrows()]

bars = ax.barh(range(len(sim_ranked)), sim_ranked['Score_pts_fin'], color=colors_c, alpha=0.85, edgecolor='white')
for i, (_, r) in enumerate(sim_ranked.iterrows()):
    flag = " ★" if r['Gagnant'] else ""
    ax.text(r['Score_pts_fin'] + 0.05, i, f"{r['Score_pts_fin']:+.2f}{flag}", fontsize=8, va='center', fontweight='bold')

ax.set_yticks(range(len(sim_ranked)))
ax.set_yticklabels(bo_labels, fontsize=8)
ax.set_xlabel('Score moyen /répondant (pts)', fontsize=11, fontweight='bold')
ax.set_title('Classement — Score Final\n★ = Gagnant du groupe', fontsize=12, fontweight='bold')
handles = [mpatches.Patch(color=colors_4[g-1], label=group_labels[g]) for g in sorted(sim_df['Groupe'].unique())]
ax.legend(handles=handles, fontsize=9, loc='lower right')

# 8b : Classement par progression (Prix Effort)
ax = axes[1]
sim_effort = sim_df.sort_values('Delta_score_pts', ascending=True)
colors_e = [colors_4[r['Groupe']-1] for _, r in sim_effort.iterrows()]
bo_labels_e = [r['BO'].replace('BO ', '') for _, r in sim_effort.iterrows()]

bars = ax.barh(range(len(sim_effort)), sim_effort['Delta_score_pts'], color=colors_e, alpha=0.85, edgecolor='white')
for i, (_, r) in enumerate(sim_effort.iterrows()):
    if r['Prix_Effort']:
        ax.text(r['Delta_score_pts'] + 0.02, i, f"{r['Delta_score_pts']:+.2f} PRIX EFFORT", fontsize=8, va='center', fontweight='bold', color='#E65100')
    else:
        ax.text(r['Delta_score_pts'] + 0.02, i, f"{r['Delta_score_pts']:+.2f}", fontsize=8, va='center', fontweight='bold')

ax.set_yticks(range(len(sim_effort)))
ax.set_yticklabels(bo_labels_e, fontsize=8)
ax.set_xlabel('Progression du score moyen /répondant (pts)', fontsize=11, fontweight='bold')
ax.set_title('Classement — Progression (Prix Effort)\nPRIX EFFORT = meilleure progression tous groupes', fontsize=12, fontweight='bold')
ax.legend(handles=handles, fontsize=9, loc='lower right')

plt.suptitle('5 Prix : 4 Gagnants de groupe + 1 Prix Effort', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/workspace/08_vue_ensemble_simulation.png', dpi=150, bbox_inches='tight')
plt.close()

# --- Fig 9 : Tableaux de résultats par groupe ---
for g in sorted(sim_df['Groupe'].unique()):
    grp = sim_df[sim_df['Groupe'] == g].sort_values('Score_pts_fin', ascending=False)
    letter = chr(64 + g)

    fig, ax = plt.subplots(figsize=(20, max(3, len(grp) * 0.9 + 2.5)))
    ax.axis('off')

    col_labels = ['BO', 'Enq\n/2m',
                  '4/4\nAvant', '3/4\nAvant', '2/4\nAvant', '1/4\nAvant', 'Rép\nAvant', 'Score\nAvant',
                  '4/4\nAprès', '3/4\nAprès', '2/4\nAprès', '1/4\nAprès', 'Rép\nAprès', 'Score\nAprès',
                  'Δ Score', 'Rang']

    table_data = []
    cell_colors = []
    for _, r in grp.iterrows():
        name = r['BO'].replace('BO ', '')
        rang_str = '★ 1er' if r['Gagnant'] else f"{r['Rang']}e"
        if r['Prix_Effort']:
            rang_str += ' EFFORT'
        row_data = [
            name, str(r['Enq_2m']),
            str(r['N4_ini']), str(r['N3_ini']), str(r['N2_ini']), str(r['N1_ini']),
            str(r['Rep_ini']), f"{r['Score_pts_ini']:+.2f}",
            str(r['N4_fin']), str(r['N3_fin']), str(r['N2_fin']), str(r['N1_fin']),
            str(r['Rep_fin']), f"{r['Score_pts_fin']:+.2f}",
            f"{r['Delta_score_pts']:+.2f}", rang_str
        ]
        table_data.append(row_data)

        if r['Gagnant']:
            row_colors = ['#FFF9C4'] * len(col_labels)
        elif r['Prix_Effort']:
            row_colors = ['#FFE0B2'] * len(col_labels)
        else:
            row_colors = ['#FAFAFA'] * len(col_labels)
        cell_colors.append(row_colors)

    table = ax.table(cellText=table_data, colLabels=col_labels, cellLoc='center',
                     loc='center', cellColours=cell_colors)
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor(colors_4[g-1])
            cell.set_text_props(color='white', fontweight='bold', fontsize=8)
            cell.set_edgecolor('white')
        else:
            cell.set_edgecolor('#E0E0E0')
            if col == 5 and row > 0:
                val = int(table_data[row-1][col])
                if val >= 3: cell.set_facecolor('#FFCDD2')
            if col == 11 and row > 0:
                val = int(table_data[row-1][col])
                if val >= 2: cell.set_facecolor('#FFCDD2')
                elif val <= 1: cell.set_facecolor('#C8E6C9')

    winner = grp.iloc[0]['BO'].replace('BO ', '')
    ax.set_title(f"{group_labels[g]} — ★ Gagnant : {winner}\n"
                 f"Classement = meilleur score moyen /répondant  |  "
                 f"4/4 = +{PTS_4}  |  3/4 = +{PTS_3}  |  2/4 = {PTS_2}  |  1/4 = {PTS_1}",
                 fontsize=11, fontweight='bold', color=colors_4[g-1], pad=20)
    plt.tight_layout()
    plt.savefig(f'/workspace/09_{letter}_tableau_resultats.png', dpi=150, bbox_inches='tight')
    plt.close()

# Export CSV
export_cols = ['BO', 'Groupe', 'Enq_2m',
               'N4_ini', 'N3_ini', 'N2_ini', 'N1_ini', 'Rep_ini', 'Pts_ini', 'Score_pts_ini',
               'N4_fin', 'N3_fin', 'N2_fin', 'N1_fin', 'Rep_fin', 'Pts_fin', 'Score_pts_fin',
               'Delta_score_pts', 'Rang', 'Gagnant', 'Prix_Effort']
sim_df[export_cols].sort_values(['Groupe', 'Score_pts_fin'], ascending=[True, False]).to_csv(
    '/workspace/simulation_resultats.csv', index=False, sep=';')

print(f"\nFichier CSV exporté: simulation_resultats.csv")
print(f"Graphiques exportés: 06, 07, 08, 09_A/B/C/D")
