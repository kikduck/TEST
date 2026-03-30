import pandas as pd
import numpy as np
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

group_labels = {
    1: "Groupe A", 2: "Groupe B", 3: "Groupe C", 4: "Groupe D"
}
colors_4 = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']

# =============================================================================
# CALCUL DES VOLUMES "AVANT CHALLENGE" (période 2 mois = annuel / 6)
# =============================================================================
df['Enq_2m'] = (df['Enquetes_annuel'] / 6).round(0).astype(int)
df['Rep_2m_ini'] = (df['Enq_2m'] * df['Tx_rep_ini'] / 100).round(0).astype(int)
df['PDTS_2m_ini'] = (df['est_PDTS_annuel'] / 6).round(0).astype(int)
df['Tx_PDTS_ini'] = np.where(df['Rep_2m_ini'] > 0,
                              (df['PDTS_2m_ini'] / df['Rep_2m_ini'] * 100).round(2), 0)

# =============================================================================
# SIMULATION RÉALISTE "APRÈS CHALLENGE" (2 mois)
# =============================================================================
# Hypothèses réalistes par BO :
# - Le challenge motive tout le monde, mais pas de manière identique
# - Les BOs avec un faible taux de répondants ont plus de marge d'amélioration
# - Les BOs avec beaucoup de PDTS peuvent en réduire davantage
# - Les BOs déjà excellentes ont moins de marge (effet plafond)
# - Le volume d'enquêtes envoyées reste stable (pas sous le contrôle des BOs)

sim = {
    # Groupe A — déjà performantes, petit volume
    # Marge de progression limitée sur le score, possible sur le taux répondants
    'BO Oyonnax':              {'delta_tx_rep': +3.5, 'pdts_fin': 0,  'effort': 'moyen'},
    'BO Aubenas-Joyeuse':      {'delta_tx_rep': +2.0, 'pdts_fin': 0,  'effort': 'bon'},
    'BO Privas-Le Cheylard':   {'delta_tx_rep': +1.0, 'pdts_fin': 1,  'effort': 'moyen'},
    'BO Crest Die':            {'delta_tx_rep': +2.5, 'pdts_fin': 1,  'effort': 'fort'},
    'BO Saint Vallier':        {'delta_tx_rep': +1.5, 'pdts_fin': 0,  'effort': 'fort'},
    'BO Valréas':              {'delta_tx_rep': +3.0, 'pdts_fin': 0,  'effort': 'très fort'},

    # Groupe B — bon score, taux répondants à améliorer (~13%)
    'BO Roussillon':           {'delta_tx_rep': +2.0, 'pdts_fin': 0,  'effort': 'bon'},
    'BO Ambérieu - Belley':    {'delta_tx_rep': +3.0, 'pdts_fin': 1,  'effort': 'fort'},
    'BO Annonay':              {'delta_tx_rep': +2.5, 'pdts_fin': 0,  'effort': 'très fort'},
    'BO Valence':              {'delta_tx_rep': +1.5, 'pdts_fin': 1,  'effort': 'moyen'},
    'BO Décines':              {'delta_tx_rep': +3.5, 'pdts_fin': 1,  'effort': 'fort'},

    # Groupe C — score intermédiaire, gros volumes, beaucoup de PDTS
    'BO Roannais':             {'delta_tx_rep': +1.5, 'pdts_fin': 2,  'effort': 'moyen'},
    'BO Forez':                {'delta_tx_rep': +2.0, 'pdts_fin': 1,  'effort': 'fort'},
    'BO Montélimar':           {'delta_tx_rep': +1.0, 'pdts_fin': 1,  'effort': 'moyen'},
    'BO Bourg - Montrevel':    {'delta_tx_rep': +2.5, 'pdts_fin': 2,  'effort': 'bon'},
    "BO l'Arbresle":           {'delta_tx_rep': +1.0, 'pdts_fin': 1,  'effort': 'faible'},
    'BO Heyrieux':             {'delta_tx_rep': +2.0, 'pdts_fin': 1,  'effort': 'fort'},
    'BO Romans':               {'delta_tx_rep': +3.0, 'pdts_fin': 1,  'effort': 'très fort'},
    'BO Gleize':               {'delta_tx_rep': +1.5, 'pdts_fin': 2,  'effort': 'moyen'},

    # Groupe D — score à renforcer, taux répondants faible
    'BO Rillieux':             {'delta_tx_rep': +4.0, 'pdts_fin': 1,  'effort': 'très fort'},
    'BO Saint Étienne':        {'delta_tx_rep': +2.0, 'pdts_fin': 2,  'effort': 'bon'},
    'BO Vénissieux':           {'delta_tx_rep': +3.0, 'pdts_fin': 1,  'effort': 'fort'},
    'BO Firminy - Saint Bonnet': {'delta_tx_rep': +2.5, 'pdts_fin': 1, 'effort': 'fort'},
    'BO Givors':               {'delta_tx_rep': +1.5, 'pdts_fin': 2,  'effort': 'moyen'},
    'BO Oullins':              {'delta_tx_rep': +2.0, 'pdts_fin': 2,  'effort': 'bon'},
}

# Appliquer la simulation
results = []
for _, row in df.iterrows():
    bo = row['BO']
    s = sim[bo]

    enq_2m = row['Enq_2m']
    tx_rep_ini = row['Tx_rep_ini']
    rep_ini = row['Rep_2m_ini']
    pdts_ini = row['PDTS_2m_ini']
    score_ini = row['Score_ini']

    tx_rep_fin = tx_rep_ini + s['delta_tx_rep']
    rep_fin = int(round(enq_2m * tx_rep_fin / 100))
    pdts_fin = s['pdts_fin']

    # Taux de PDTS (en % des répondants)
    tx_pdts_ini = (pdts_ini / rep_ini * 100) if rep_ini > 0 else 0
    tx_pdts_fin = (pdts_fin / rep_fin * 100) if rep_fin > 0 else 0

    # Nouveaux répondants gagnés
    new_rep = rep_fin - rep_ini

    # Simulation réaliste de la répartition des notes sur la période challenge
    # Les nouveaux répondants se répartissent avec un biais positif (effet challenge)
    # et les anciens PDTS sont réduits
    # Score fin ≈ score_ini + effet de la réduction PDTS + effet des nouveaux répondants
    # On modélise : chaque PDTS réduit apporte ~(100/rep_fin)% de score
    #               chaque nouveau répondant satisfait apporte aussi
    pdts_reduits = pdts_ini - pdts_fin
    if rep_fin > 0:
        impact_pdts = pdts_reduits / rep_fin * 100
        impact_new_rep = new_rep * 0.85 / rep_fin * 100 if new_rep > 0 else 0
        score_fin = score_ini + impact_pdts + impact_new_rep * 0.5
        score_fin = min(score_fin, 98.0)
        score_fin = round(score_fin, 2)
    else:
        score_fin = score_ini

    results.append({
        'BO': bo,
        'Groupe': row['Groupe'],
        'Enq_2m': enq_2m,

        'Rep_ini': rep_ini,
        'Tx_rep_ini': round(tx_rep_ini, 2),
        'PDTS_ini': pdts_ini,
        'Tx_PDTS_ini': round(tx_pdts_ini, 2),
        'Score_ini': score_ini,

        'Rep_fin': rep_fin,
        'Tx_rep_fin': round(tx_rep_fin, 2),
        'PDTS_fin': pdts_fin,
        'Tx_PDTS_fin': round(tx_pdts_fin, 2),
        'Score_fin': score_fin,

        'Delta_score': round(score_fin - score_ini, 2),
        'Delta_tx_rep': round(tx_rep_fin - tx_rep_ini, 2),
        'Delta_tx_PDTS': round(tx_pdts_ini - tx_pdts_fin, 2),
        'Effort': s['effort']
    })

sim_df = pd.DataFrame(results)

# =============================================================================
# CALCUL DU SCORING CHALLENGE (100 pts)
# =============================================================================
def calc_scoring(group_df):
    scored = group_df.copy()

    # 1. Progression Score (40 pts)
    max_delta_score = scored['Delta_score'].max()
    if max_delta_score > 0:
        scored['Pts_score'] = (scored['Delta_score'].clip(lower=0) / max_delta_score * 40).round(1)
    else:
        scored['Pts_score'] = 0

    # 2. Réduction Taux PDTS (30 pts)
    max_delta_pdts = scored['Delta_tx_PDTS'].max()
    if max_delta_pdts > 0:
        scored['Pts_PDTS'] = (scored['Delta_tx_PDTS'].clip(lower=0) / max_delta_pdts * 30).round(1)
    else:
        scored['Pts_PDTS'] = 0

    # 3. Progression Taux Répondants (20 pts)
    max_delta_rep = scored['Delta_tx_rep'].max()
    if max_delta_rep > 0:
        scored['Pts_rep'] = (scored['Delta_tx_rep'].clip(lower=0) / max_delta_rep * 20).round(1)
    else:
        scored['Pts_rep'] = 0

    # 4. Bonus Excellence (10 pts)
    scored['Pts_bonus'] = 0.0
    scored.loc[scored['Score_fin'] >= 90, 'Pts_bonus'] += 5
    scored.loc[scored['Tx_rep_fin'] >= 18, 'Pts_bonus'] += 5

    scored['TOTAL'] = (scored['Pts_score'] + scored['Pts_PDTS'] +
                       scored['Pts_rep'] + scored['Pts_bonus']).round(1)

    return scored

all_scored = []
for g in sorted(sim_df['Groupe'].unique()):
    grp = sim_df[sim_df['Groupe'] == g].copy()
    scored = calc_scoring(grp)
    all_scored.append(scored)

sim_df = pd.concat(all_scored, ignore_index=True)
sim_df['Rang'] = sim_df.groupby('Groupe')['TOTAL'].rank(ascending=False, method='min').astype(int)
sim_df['Gagnant'] = sim_df['Rang'] == 1

# =============================================================================
# AFFICHAGE CONSOLE
# =============================================================================
print("=" * 100)
print("SIMULATION DU CHALLENGE — RÉSULTATS AVANT / APRÈS (période 2 mois)")
print("=" * 100)

for g in sorted(sim_df['Groupe'].unique()):
    grp = sim_df[sim_df['Groupe'] == g].sort_values('TOTAL', ascending=False)
    letter = chr(64 + g)
    n = len(grp)
    winner = grp.iloc[0]['BO']

    print(f"\n{'━' * 100}")
    print(f"  {group_labels[g]} ({n} BOs) — GAGNANT : {winner}")
    print(f"{'━' * 100}")
    print()
    print(f"  {'BO':<28} │ {'AVANT CHALLENGE':^30} │ {'APRÈS CHALLENGE':^30} │ {'SCORING':^25}")
    print(f"  {'':<28} │ {'Score':>7} {'TxRép':>7} {'PDTS':>5} {'Rép':>5} │ {'Score':>7} {'TxRép':>7} {'PDTS':>5} {'Rép':>5} │ {'Sc':>4} {'PD':>4} {'Rp':>4} {'Bn':>4} {'TOT':>5} {'Rg':>3}")
    print(f"  {'─' * 28}─┼─{'─' * 30}─┼─{'─' * 30}─┼─{'─' * 25}")

    for _, r in grp.iterrows():
        name = r['BO'].replace('BO ', '')
        flag = " ★" if r['Gagnant'] else "  "
        print(f"  {name:<28} │ "
              f"{r['Score_ini']:>6.1f}% {r['Tx_rep_ini']:>6.1f}% {r['PDTS_ini']:>4}  {r['Rep_ini']:>4} │ "
              f"{r['Score_fin']:>6.1f}% {r['Tx_rep_fin']:>6.1f}% {r['PDTS_fin']:>4}  {r['Rep_fin']:>4} │ "
              f"{r['Pts_score']:>4.0f} {r['Pts_PDTS']:>4.0f} {r['Pts_rep']:>4.0f} {r['Pts_bonus']:>4.0f} {r['TOTAL']:>5.1f}{flag}")

    print()
    avg_before = grp['Score_ini'].mean()
    avg_after = grp['Score_fin'].mean()
    print(f"  Moyenne groupe : Score {avg_before:.1f}% → {avg_after:.1f}% (Δ +{avg_after-avg_before:.1f}%)")
    print(f"  Taux Rép. moy. : {grp['Tx_rep_ini'].mean():.1f}% → {grp['Tx_rep_fin'].mean():.1f}%")
    print(f"  PDTS total     : {grp['PDTS_ini'].sum()} → {grp['PDTS_fin'].sum()}")

print("\n\n" + "=" * 100)
print("PODIUM DES GAGNANTS")
print("=" * 100)
for g in sorted(sim_df['Groupe'].unique()):
    w = sim_df[(sim_df['Groupe'] == g) & (sim_df['Gagnant'])].iloc[0]
    print(f"  {group_labels[g]:>10} :  {w['BO']:<30}  {w['TOTAL']:.1f} pts  "
          f"(Score {w['Score_ini']:.1f}% → {w['Score_fin']:.1f}%, "
          f"PDTS {w['PDTS_ini']} → {w['PDTS_fin']}, "
          f"TxRép {w['Tx_rep_ini']:.1f}% → {w['Tx_rep_fin']:.1f}%)")

# =============================================================================
# VISUALISATIONS
# =============================================================================
sns.set_theme(style="whitegrid", font_scale=1.0)

# --- Fig 6 : Avant / Après Score par BO, trié par groupe puis par score final ---
fig, ax = plt.subplots(figsize=(20, 12))

sim_sorted = sim_df.sort_values(['Groupe', 'TOTAL'], ascending=[True, False])
y_pos = list(range(len(sim_sorted)))
bo_labels = [r['BO'].replace('BO ', '') for _, r in sim_sorted.iterrows()]

bar_height = 0.35
for i, (_, r) in enumerate(sim_sorted.iterrows()):
    color = colors_4[r['Groupe']-1]

    ax.barh(i + bar_height/2, r['Score_ini'], height=bar_height,
            color=color, alpha=0.35, edgecolor=color, linewidth=0.8)
    ax.barh(i - bar_height/2, r['Score_fin'], height=bar_height,
            color=color, alpha=0.85, edgecolor='white', linewidth=0.8)

    ax.text(r['Score_fin'] + 0.3, i - bar_height/2,
            f"{r['Score_fin']:.1f}%", va='center', fontsize=7.5, fontweight='bold', color=color)
    ax.text(r['Score_ini'] + 0.3, i + bar_height/2,
            f"{r['Score_ini']:.1f}%", va='center', fontsize=7.5, color='gray')

    delta = r['Delta_score']
    ax.text(max(r['Score_fin'], r['Score_ini']) + 4, i,
            f"+{delta:.1f}%" if delta > 0 else f"{delta:.1f}%",
            va='center', fontsize=8, fontweight='bold',
            color='#2E7D32' if delta > 0 else '#C62828')

    if r['Gagnant']:
        ax.text(68, i, "★ GAGNANT", va='center', fontsize=8, fontweight='bold',
                color='#FFD600', bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.9))

# Séparations entre groupes
current_g = None
sep_positions = []
for i, (_, r) in enumerate(sim_sorted.iterrows()):
    if current_g is not None and r['Groupe'] != current_g:
        sep_positions.append(i - 0.5)
    current_g = r['Groupe']
for sp in sep_positions:
    ax.axhline(y=sp, color='gray', linewidth=1.5, linestyle='--', alpha=0.5)

ax.set_yticks(y_pos)
ax.set_yticklabels(bo_labels, fontsize=9)
ax.set_xlabel('Score de Satisfaction (%)', fontsize=12, fontweight='bold')
ax.set_xlim(65, 105)
ax.set_title('Simulation Challenge — Score AVANT (pâle) vs APRÈS (foncé) par BO\n★ = Gagnant du groupe',
             fontsize=14, fontweight='bold')
ax.invert_yaxis()

handles = []
for g in sorted(sim_df['Groupe'].unique()):
    handles.append(mpatches.Patch(color=colors_4[g-1], alpha=0.85, label=f"{group_labels[g]}"))
handles.append(mpatches.Patch(color='gray', alpha=0.35, label="Avant challenge"))
handles.append(mpatches.Patch(color='gray', alpha=0.85, label="Après challenge"))
ax.legend(handles=handles, loc='lower right', fontsize=9)

plt.tight_layout()
plt.savefig('/workspace/06_simulation_avant_apres.png', dpi=150, bbox_inches='tight')
plt.close()

# --- Fig 7 : Décomposition du scoring par BO ---
fig, axes = plt.subplots(2, 2, figsize=(20, 16))

for idx, g in enumerate(sorted(sim_df['Groupe'].unique())):
    ax = axes[idx // 2][idx % 2]
    grp = sim_df[sim_df['Groupe'] == g].sort_values('TOTAL', ascending=True)
    color = colors_4[g-1]

    bo_short = [bo.replace('BO ', '') for bo in grp['BO']]
    y = range(len(grp))

    bars_score = ax.barh(y, grp['Pts_score'], height=0.6, color='#1565C0', alpha=0.8, label='Progression Score (40)')
    left = grp['Pts_score'].values
    bars_pdts = ax.barh(y, grp['Pts_PDTS'], height=0.6, left=left, color='#E65100', alpha=0.8, label='Réduction PDTS (30)')
    left = left + grp['Pts_PDTS'].values
    bars_rep = ax.barh(y, grp['Pts_rep'], height=0.6, left=left, color='#2E7D32', alpha=0.8, label='Taux Répondants (20)')
    left = left + grp['Pts_rep'].values
    bars_bonus = ax.barh(y, grp['Pts_bonus'], height=0.6, left=left, color='#FFD600', alpha=0.9, label='Bonus Excellence (10)')

    for i, (_, r) in enumerate(grp.iterrows()):
        total = r['TOTAL']
        flag = " ★" if r['Gagnant'] else ""
        ax.text(total + 0.5, i, f"{total:.1f} pts{flag}", va='center', fontsize=9, fontweight='bold')

    ax.set_yticks(list(y))
    ax.set_yticklabels(bo_short, fontsize=10)
    ax.set_xlabel('Points Challenge', fontsize=10)
    ax.set_title(f"{group_labels[g]} ({len(grp)} BOs)", fontsize=12, fontweight='bold', color=color)
    ax.set_xlim(0, 110)
    ax.legend(fontsize=7, loc='lower right')

plt.suptitle('Décomposition du Score Challenge par Groupe\n(40 pts Score + 30 pts PDTS + 20 pts Répondants + 10 pts Bonus)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/workspace/07_decomposition_scoring.png', dpi=150, bbox_inches='tight')
plt.close()

# --- Fig 8 : Vue d'ensemble Avant/Après par métrique ---
fig, axes = plt.subplots(1, 3, figsize=(20, 8))

# 8a: Score avant/après
ax = axes[0]
for g in sorted(sim_df['Groupe'].unique()):
    grp = sim_df[sim_df['Groupe'] == g]
    for _, r in grp.iterrows():
        name = r['BO'].replace('BO ', '')
        ax.annotate('', xy=(r['Score_fin'], r['Tx_rep_fin']),
                    xytext=(r['Score_ini'], r['Tx_rep_ini']),
                    arrowprops=dict(arrowstyle='->', color=colors_4[g-1], lw=1.5, alpha=0.7))
        ax.scatter(r['Score_ini'], r['Tx_rep_ini'], s=40, color=colors_4[g-1], alpha=0.3, zorder=3)
        ax.scatter(r['Score_fin'], r['Tx_rep_fin'], s=80, color=colors_4[g-1], alpha=0.9, 
                   edgecolors='white', linewidth=0.8, zorder=4)
        ax.annotate(name, (r['Score_fin'], r['Tx_rep_fin']),
                    fontsize=6, ha='center', va='bottom', xytext=(0, 4), textcoords='offset points')

ax.set_xlabel('Score Satisfaction (%)', fontsize=11, fontweight='bold')
ax.set_ylabel('Taux de Répondants (%)', fontsize=11, fontweight='bold')
ax.set_title('Trajectoire Score × Taux Répondants\n(point pâle = avant, flèche → après)', fontsize=11, fontweight='bold')
handles = [mpatches.Patch(color=colors_4[g-1], label=group_labels[g]) for g in sorted(sim_df['Groupe'].unique())]
ax.legend(handles=handles, fontsize=8)

# 8b: Réduction PDTS par BO
ax = axes[1]
sim_sorted2 = sim_df.sort_values(['Groupe', 'Delta_tx_PDTS'], ascending=[True, False])
bo_labels2 = [r['BO'].replace('BO ', '') for _, r in sim_sorted2.iterrows()]
colors_b = [colors_4[r['Groupe']-1] for _, r in sim_sorted2.iterrows()]

bars = ax.barh(range(len(sim_sorted2)), sim_sorted2['PDTS_ini'], height=0.4,
               color=[c for c in colors_b], alpha=0.3, label='PDTS Avant')
ax.barh([i-0.0 for i in range(len(sim_sorted2))], sim_sorted2['PDTS_fin'], height=0.4,
        color=[c for c in colors_b], alpha=0.85, label='PDTS Après')

for i, (_, r) in enumerate(sim_sorted2.iterrows()):
    if r['PDTS_ini'] > 0:
        ax.text(r['PDTS_ini'] + 0.1, i, f"{r['PDTS_ini']}→{r['PDTS_fin']}", fontsize=7, va='center')

ax.set_yticks(range(len(sim_sorted2)))
ax.set_yticklabels(bo_labels2, fontsize=7.5)
ax.set_xlabel('Nombre de PDTS (sur 2 mois)', fontsize=11, fontweight='bold')
ax.set_title('Réduction des PDTS\n(pâle = avant, foncé = après)', fontsize=11, fontweight='bold')
ax.invert_yaxis()

# 8c: Total scoring par BO
ax = axes[2]
sim_ranked = sim_df.sort_values('TOTAL', ascending=True)
bo_labels3 = [r['BO'].replace('BO ', '') for _, r in sim_ranked.iterrows()]
colors_c = [colors_4[r['Groupe']-1] for _, r in sim_ranked.iterrows()]

bars = ax.barh(range(len(sim_ranked)), sim_ranked['TOTAL'], color=colors_c, alpha=0.85, edgecolor='white')
for i, (_, r) in enumerate(sim_ranked.iterrows()):
    flag = " ★" if r['Gagnant'] else ""
    ax.text(r['TOTAL'] + 0.5, i, f"{r['TOTAL']:.1f}{flag}", fontsize=7.5, va='center', fontweight='bold')

ax.set_yticks(range(len(sim_ranked)))
ax.set_yticklabels(bo_labels3, fontsize=7.5)
ax.set_xlabel('Score Challenge (sur 100)', fontsize=11, fontweight='bold')
ax.set_title('Classement Général\n(★ = gagnant du groupe)', fontsize=11, fontweight='bold')

plt.suptitle('Simulation Challenge — Vue d\'ensemble Avant / Après', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/workspace/08_vue_ensemble_simulation.png', dpi=150, bbox_inches='tight')
plt.close()

# --- Fig 9 : Tableau de résultats complet par groupe ---
for g in sorted(sim_df['Groupe'].unique()):
    grp = sim_df[sim_df['Groupe'] == g].sort_values('TOTAL', ascending=False)
    letter = chr(64 + g)

    fig, ax = plt.subplots(figsize=(18, max(3, len(grp) * 0.9 + 2.5)))
    ax.axis('off')

    col_labels = ['BO', 'Enq.\n/2m',
                  'Score\nAvant', 'Score\nAprès', 'Δ Score',
                  'Tx Rép\nAvant', 'Tx Rép\nAprès', 'Δ Tx Rép',
                  'PDTS\nAvant', 'PDTS\nAprès',
                  'Pts\nScore', 'Pts\nPDTS', 'Pts\nRép', 'Bonus', 'TOTAL', 'Rang']

    table_data = []
    cell_colors = []
    for _, r in grp.iterrows():
        name = r['BO'].replace('BO ', '')
        row_data = [
            name, str(r['Enq_2m']),
            f"{r['Score_ini']:.1f}%", f"{r['Score_fin']:.1f}%", f"+{r['Delta_score']:.1f}%",
            f"{r['Tx_rep_ini']:.1f}%", f"{r['Tx_rep_fin']:.1f}%", f"+{r['Delta_tx_rep']:.1f}%",
            str(r['PDTS_ini']), str(r['PDTS_fin']),
            f"{r['Pts_score']:.0f}", f"{r['Pts_PDTS']:.0f}", f"{r['Pts_rep']:.0f}",
            f"{r['Pts_bonus']:.0f}", f"{r['TOTAL']:.1f}", f"{'★ 1er' if r['Gagnant'] else r['Rang']}"
        ]
        table_data.append(row_data)

        base_color = colors_4[g-1]
        if r['Gagnant']:
            row_colors = ['#FFF9C4'] * len(col_labels)
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
        if col >= 10 and row > 0:
            cell.set_facecolor('#F3E5F5' if not table_data[row-1][-1].startswith('★') else '#FFF9C4')

    ax.set_title(f"Résultats Simulation — {group_labels[g]}",
                 fontsize=14, fontweight='bold', color=colors_4[g-1], pad=20)
    plt.tight_layout()
    plt.savefig(f'/workspace/09_{letter}_tableau_resultats.png', dpi=150, bbox_inches='tight')
    plt.close()

# Export CSV complet
export_cols = ['BO', 'Groupe', 'Enq_2m',
               'Score_ini', 'Score_fin', 'Delta_score',
               'Tx_rep_ini', 'Tx_rep_fin', 'Delta_tx_rep',
               'PDTS_ini', 'PDTS_fin', 'Tx_PDTS_ini', 'Tx_PDTS_fin', 'Delta_tx_PDTS',
               'Rep_ini', 'Rep_fin',
               'Pts_score', 'Pts_PDTS', 'Pts_rep', 'Pts_bonus', 'TOTAL', 'Rang', 'Gagnant']
sim_df[export_cols].sort_values(['Groupe', 'TOTAL'], ascending=[True, False]).to_csv(
    '/workspace/simulation_resultats.csv', index=False, sep=';')

print(f"\nFichier CSV exporté: simulation_resultats.csv")
print(f"Graphiques exportés: 06, 07, 08, 09_A/B/C/D")
