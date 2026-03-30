from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")

ROOT = Path("/workspace")
CHALLENGE_MONTHS = 2
N_GROUPS = 4


# =============================================================================
# DONNEES ANNUELLES 2025
# =============================================================================
data = {
    "Base opérationnelle": [
        "BO Forez",
        "BO Roannais",
        "BO Aubenas-Joyeuse",
        "BO Ambérieu - Belley",
        "BO Bourg - Montrevel",
        "BO Montélimar",
        "BO Gleize",
        "BO Heyrieux",
        "BO Roussillon",
        "BO l'Arbresle",
        "BO Crest Die",
        "BO Valence",
        "BO Privas-Le Cheylard",
        "BO Romans",
        "BO Décines",
        "BO Saint Vallier",
        "BO Saint Étienne",
        "BO Givors",
        "BO Oullins",
        "BO Annonay",
        "BO Valréas",
        "BO Rillieux",
        "BO Vénissieux",
        "BO Firminy - Saint Bonnet",
        "BO Oyonnax",
    ],
    "Taux composite cumulé": [
        3.93,
        4.63,
        1.68,
        3.24,
        10.91,
        6.05,
        7.16,
        9.13,
        6.41,
        8.90,
        2.89,
        4.04,
        9.40,
        14.61,
        3.08,
        4.81,
        11.47,
        12.85,
        12.24,
        3.04,
        11.77,
        4.44,
        12.98,
        8.05,
        0.00,
    ],
    "Nouveau Score Satisfaction": [
        85.02,
        85.31,
        90.10,
        88.89,
        80.49,
        82.91,
        78.39,
        78.61,
        90.00,
        80.00,
        87.41,
        85.52,
        87.86,
        78.43,
        84.96,
        85.61,
        77.14,
        73.91,
        72.14,
        88.07,
        83.33,
        78.95,
        76.04,
        74.42,
        95.00,
    ],
    "Poids Global": [
        1.28,
        1.56,
        0.20,
        0.45,
        1.71,
        0.89,
        1.56,
        1.56,
        0.27,
        0.89,
        0.55,
        0.66,
        0.45,
        1.01,
        0.55,
        0.55,
        1.28,
        1.71,
        1.71,
        0.27,
        0.36,
        0.55,
        0.77,
        0.77,
        0.00,
    ],
    "Nouveau Poids Global": [
        1.74,
        1.73,
        1.30,
        1.25,
        1.21,
        1.21,
        1.12,
        1.02,
        0.99,
        0.87,
        0.80,
        0.79,
        0.78,
        0.75,
        0.69,
        0.69,
        0.64,
        0.59,
        0.58,
        0.54,
        0.41,
        0.37,
        0.36,
        0.29,
        0.13,
    ],
    "est_PDTS": [
        14,
        16,
        4,
        7,
        17,
        11,
        16,
        16,
        5,
        11,
        8,
        9,
        7,
        12,
        8,
        8,
        14,
        17,
        17,
        5,
        6,
        8,
        10,
        10,
        0,
    ],
    "Nombre enquêtes envoyées": [
        1609,
        1411,
        1130,
        1353,
        1489,
        1251,
        1324,
        1106,
        1229,
        1126,
        922,
        1083,
        755,
        990,
        987,
        817,
        1157,
        1098,
        1064,
        844,
        571,
        873,
        785,
        566,
        285,
    ],
    "Taux répondants cumulé": [
        15.35,
        17.36,
        16.99,
        13.97,
        13.77,
        15.91,
        15.03,
        16.91,
        13.02,
        14.65,
        15.51,
        13.39,
        18.54,
        15.45,
        13.48,
        16.16,
        12.10,
        12.57,
        13.16,
        12.91,
        16.81,
        10.88,
        12.23,
        15.19,
        14.04,
    ],
}

df = pd.DataFrame(data)


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(value, max_value))


def fmt_pct(value: float) -> str:
    return f"{value:.1f}".replace(".", ",") + " %"


def fmt_num(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.1f}".replace(".", ",")


def compute_baseline_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    baseline = frame.copy()
    baseline["Nombre répondants estimé annuel"] = (
        baseline["Nombre enquêtes envoyées"] * baseline["Taux répondants cumulé"] / 100
    ).round(0).astype(int)
    baseline["Taux PDTS actuel"] = np.where(
        baseline["Nombre répondants estimé annuel"] > 0,
        baseline["est_PDTS"] / baseline["Nombre répondants estimé annuel"] * 100,
        0.0,
    )
    baseline["Enquêtes / mois"] = (baseline["Nombre enquêtes envoyées"] / 12).round(0).astype(int)
    baseline["Enquêtes / 2 mois"] = (
        baseline["Nombre enquêtes envoyées"] / (12 / CHALLENGE_MONTHS)
    ).round(0).astype(int)
    baseline["Répondants / mois"] = (
        baseline["Nombre répondants estimé annuel"] / 12
    ).round(0).astype(int)
    baseline["Répondants / 2 mois"] = (
        baseline["Nombre répondants estimé annuel"] / (12 / CHALLENGE_MONTHS)
    ).round(0).astype(int)
    baseline["PDTS / mois"] = (baseline["est_PDTS"] / 12).round(1)
    baseline["PDTS / 2 mois"] = (baseline["est_PDTS"] / (12 / CHALLENGE_MONTHS)).round(1)
    return baseline


def assign_groups(frame: pd.DataFrame) -> tuple[pd.DataFrame, str, float, float]:
    features_for_clustering = [
        "Nombre enquêtes envoyées",
        "Taux répondants cumulé",
        "est_PDTS",
        "Nouveau Score Satisfaction",
    ]
    X = frame[features_for_clustering].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans_final = KMeans(n_clusters=N_GROUPS, random_state=42, n_init=30)
    labels_kmeans = kmeans_final.fit_predict(X_scaled)
    sil_km = silhouette_score(X_scaled, labels_kmeans)

    agg = AgglomerativeClustering(n_clusters=N_GROUPS, linkage="ward")
    labels_agg = agg.fit_predict(X_scaled)
    sil_agg = silhouette_score(X_scaled, labels_agg)

    if sil_agg > sil_km:
        labels_best = labels_agg
        method = "Hiérarchique (Ward)"
    else:
        labels_best = labels_kmeans
        method = "K-Means"

    result = frame.copy()
    result["Groupe_raw"] = labels_best
    group_means = (
        result.groupby("Groupe_raw")["Nouveau Score Satisfaction"].mean().sort_values(ascending=False)
    )
    label_mapping = {old: new + 1 for new, old in enumerate(group_means.index)}
    result["Groupe"] = result["Groupe_raw"].map(label_mapping)

    group_names = {}
    for group_id in sorted(result["Groupe"].unique()):
        n = len(result[result["Groupe"] == group_id])
        group_names[group_id] = f"Groupe {chr(64 + group_id)} ({n} BOs)"
    result["Nom Groupe"] = result["Groupe"].map(group_names)
    return result, method, sil_km, sil_agg


def build_simulation(frame: pd.DataFrame) -> pd.DataFrame:
    simulated = frame.copy()

    volume_growth_by_group = {1: 0.02, 2: 0.04, 3: 0.03, 4: 0.05}
    response_gain_by_group = {1: 0.6, 2: 1.8, 3: 1.4, 4: 2.2}
    pdts_reduction_by_group = {1: 0.08, 2: 0.14, 3: 0.20, 4: 0.24}
    score_gain_by_group = {1: 0.8, 2: 1.6, 3: 2.2, 4: 2.9}

    current_pdts_rate = simulated["Taux PDTS actuel"]
    simulated["Volume challenge / 2 mois"] = simulated.apply(
        lambda row: int(
            round(
                row["Enquêtes / 2 mois"]
                * (
                    1
                    + volume_growth_by_group[row["Groupe"]]
                    + (
                        0.02
                        if row["Nouveau Score Satisfaction"] < 80
                        else 0.01
                        if row["Taux répondants cumulé"] < 13
                        else 0.00
                    )
                    - (0.01 if row["Enquêtes / 2 mois"] > 220 else 0.00)
                )
            )
        ),
        axis=1,
    )

    simulated["Taux répondants simulé"] = simulated.apply(
        lambda row: round(
            clamp(
                row["Taux répondants cumulé"]
                + response_gain_by_group[row["Groupe"]]
                + (0.6 if row["Taux répondants cumulé"] < 13 else 0.0)
                + (0.3 if row["Nouveau Score Satisfaction"] < 78 else 0.0)
                - (0.4 if row["Taux répondants cumulé"] > 17.5 else 0.0),
                11.0,
                19.5,
            ),
            2,
        ),
        axis=1,
    )

    simulated["Répondants simulés / 2 mois"] = (
        simulated["Volume challenge / 2 mois"] * simulated["Taux répondants simulé"] / 100
    ).round(0).astype(int)

    simulated["Taux PDTS simulé"] = simulated.apply(
        lambda row: round(
            clamp(
                row["Taux PDTS actuel"]
                * (
                    1
                    - pdts_reduction_by_group[row["Groupe"]]
                    - (0.05 if row["Taux PDTS actuel"] > 9 else 0.0)
                    - (0.03 if row["Nouveau Score Satisfaction"] < 78 else 0.0)
                    + (0.04 if row["Taux PDTS actuel"] < 3 else 0.0)
                ),
                0.0,
                max(row["Taux PDTS actuel"] - 0.2, 0.0),
            ),
            2,
        ),
        axis=1,
    )

    simulated["PDTS simulés / 2 mois"] = simulated.apply(
        lambda row: int(
            round(
                max(
                    row["Répondants simulés / 2 mois"] * row["Taux PDTS simulé"] / 100,
                    0,
                )
            )
        ),
        axis=1,
    )

    simulated["Gain score simulé"] = simulated.apply(
        lambda row: round(
            clamp(
                score_gain_by_group[row["Groupe"]]
                + 0.18 * (row["Taux répondants simulé"] - row["Taux répondants cumulé"])
                + 0.22 * (row["Taux PDTS actuel"] - row["Taux PDTS simulé"])
                + (0.5 if row["Nouveau Score Satisfaction"] < 78 else 0.0)
                - (0.4 if row["Nouveau Score Satisfaction"] > 90 else 0.0),
                0.3,
                5.5,
            ),
            2,
        ),
        axis=1,
    )
    simulated["Score satisfaction simulé"] = (
        simulated["Nouveau Score Satisfaction"] + simulated["Gain score simulé"]
    ).round(2).clip(upper=96.5)

    simulated["Gain taux répondants"] = (
        simulated["Taux répondants simulé"] - simulated["Taux répondants cumulé"]
    ).round(2)
    simulated["Baisse taux PDTS"] = (
        simulated["Taux PDTS actuel"] - simulated["Taux PDTS simulé"]
    ).round(2)
    simulated["Baisse PDTS volume"] = (
        simulated["PDTS / 2 mois"] - simulated["PDTS simulés / 2 mois"]
    ).round(1)
    return simulated


def score_challenge(frame: pd.DataFrame) -> pd.DataFrame:
    scored = frame.copy()
    scored["Points Score"] = 0.0
    scored["Points PDTS"] = 0.0
    scored["Points Répondants"] = 0.0
    scored["Bonus Excellence"] = 0.0

    for group_id in sorted(scored["Groupe"].unique()):
        mask = scored["Groupe"] == group_id
        max_gain_score = scored.loc[mask, "Gain score simulé"].max()
        max_gain_pdts = scored.loc[mask, "Baisse taux PDTS"].max()
        max_gain_response = scored.loc[mask, "Gain taux répondants"].max()

        if max_gain_score > 0:
            scored.loc[mask, "Points Score"] = (
                scored.loc[mask, "Gain score simulé"] / max_gain_score * 40
            ).round(1)
        if max_gain_pdts > 0:
            scored.loc[mask, "Points PDTS"] = (
                scored.loc[mask, "Baisse taux PDTS"] / max_gain_pdts * 30
            ).round(1)
        if max_gain_response > 0:
            scored.loc[mask, "Points Répondants"] = (
                scored.loc[mask, "Gain taux répondants"] / max_gain_response * 20
            ).round(1)

    scored["Bonus Excellence"] = (
        (scored["Score satisfaction simulé"] >= 90).astype(int) * 5
        + (scored["Taux répondants simulé"] >= 18).astype(int) * 5
    )
    scored["Score challenge simulé"] = (
        scored["Points Score"]
        + scored["Points PDTS"]
        + scored["Points Répondants"]
        + scored["Bonus Excellence"]
    ).round(1)
    scored["Classement groupe"] = (
        scored.groupby("Groupe")["Score challenge simulé"].rank(method="dense", ascending=False).astype(int)
    )
    return scored


def render_markdown(frame: pd.DataFrame, simulation_assumptions: dict[int, str]) -> str:
    label_to_profile = {
        1: "Performantes / petit-moyen volume",
        2: "Performantes / moyen-grand volume",
        3: "Intermédiaires / grand volume",
        4: "En progression / volume moyen",
    }
    group_objectives = {
        1: "Sécuriser l'excellence avec une animation légère et une vigilance sur les PDTS résiduels.",
        2: "Aller chercher de la réponse utile en gardant un niveau de service déjà élevé.",
        3: "Travailler en priorité la réduction des PDTS sur des volumes significatifs.",
        4: "Activer à la fois le taux de réponse et le traitement des irritants pour faire bouger le score.",
    }

    lines = [
        "# Challenge Satisfaction Client — Simulation réaliste taux / volumes par BO",
        "",
        "## Logique de simulation retenue",
        "",
        f"Projection sur **{CHALLENGE_MONTHS} mois** à partir des données 2025 par BO, avec des hypothèses métier simples :",
        "",
        "- **Volume challenge** : base historique ramenée sur 2 mois, ajustée de la saisonnalité et du potentiel d'activation local.",
        "- **Taux de répondants simulé** : progression plus forte pour les groupes B et D, plus modérée pour les BO déjà performantes.",
        "- **Taux de PDTS simulé** : baisse proportionnelle au niveau de départ, plus ambitieuse sur les groupes C et D.",
        "- **Score satisfaction simulé** : combinaison cohérente des gains de répondants et de la baisse des PDTS, plafonnée pour rester réaliste.",
        "",
        "Les simulations ci-dessous donnent un scénario de challenge **crédible et comparable**, pas une prévision contractuelle.",
        "",
        "---",
        "",
        "## Hypothèses par groupe",
        "",
        "| Groupe | Hypothèse dominante |",
        "|---|---|",
    ]

    for group_id in sorted(simulation_assumptions):
        lines.append(f"| **{chr(64 + group_id)}** | {simulation_assumptions[group_id]} |")

    lines.extend(["", "---", "", "## Résultats simulés par groupe", ""])

    for group_id in sorted(frame["Groupe"].unique()):
        sub = frame[frame["Groupe"] == group_id].sort_values(
            ["Classement groupe", "Score challenge simulé", "Score satisfaction simulé"],
            ascending=[True, False, False],
        )
        group_name = f"Groupe {chr(64 + group_id)}"
        lines.extend(
            [
                f"### {group_name} — {label_to_profile[group_id]} ({len(sub)} BOs)",
                "",
                f"> **Enjeu** : {group_objectives[group_id]}",
                "",
                "| BO | Enquêtes /2m simulées | Tx rép. simulé | PDTS /2m simulés | Score simulé | Score challenge | Rang |",
                "|---|---|---|---|---|---|---|",
            ]
        )
        for _, row in sub.iterrows():
            lines.append(
                "| "
                + f"**{row['Base opérationnelle']}** | "
                + f"{fmt_num(row['Volume challenge / 2 mois'])} | "
                + f"{fmt_pct(row['Taux répondants simulé'])} | "
                + f"{fmt_num(row['PDTS simulés / 2 mois'])} | "
                + f"{fmt_pct(row['Score satisfaction simulé'])} | "
                + f"{fmt_num(row['Score challenge simulé'])} pts | "
                + f"{int(row['Classement groupe'])} |"
            )

        mean_volume = sub["Volume challenge / 2 mois"].mean()
        mean_response = sub["Taux répondants simulé"].mean()
        mean_pdts = sub["PDTS simulés / 2 mois"].mean()
        mean_score = sub["Score satisfaction simulé"].mean()
        best = sub.iloc[0]
        lines.extend(
            [
                "",
                "| Synthèse groupe | Valeur |",
                "|---|---|",
                f"| Volume moyen simulé /2 mois | **{fmt_num(round(mean_volume, 1))}** |",
                f"| Taux de répondants moyen simulé | **{fmt_pct(round(mean_response, 1))}** |",
                f"| PDTS moyen simulé /2 mois | **{fmt_num(round(mean_pdts, 1))}** |",
                f"| Score satisfaction moyen simulé | **{fmt_pct(round(mean_score, 1))}** |",
                f"| Leader simulé | **{best['Base opérationnelle']}** ({fmt_num(best['Score challenge simulé'])} pts) |",
                "",
                "---",
                "",
            ]
        )

    winners = (
        frame.sort_values(["Groupe", "Classement groupe", "Score challenge simulé"], ascending=[True, True, False])
        .groupby("Groupe")
        .head(1)
    )
    lines.extend(
        [
            "## Gagnants simulés du challenge",
            "",
            "| Groupe | BO gagnante simulée | Score challenge | Score satisfaction fin |",
            "|---|---|---|---|",
        ]
    )
    for _, row in winners.iterrows():
        lines.append(
            f"| **{chr(64 + row['Groupe'])}** | **{row['Base opérationnelle']}** | "
            f"{fmt_num(row['Score challenge simulé'])} pts | {fmt_pct(row['Score satisfaction simulé'])} |"
        )

    lines.extend(
        [
            "",
            "## Fichiers produits",
            "",
            "| Fichier | Description |",
            "|---|---|",
            "| `analyse_groupes_bo.py` | Script d'analyse, de simulation et d'export |",
            "| `groupes_bo_comparables.csv` | Données consolidées avec hypothèses et résultats simulés |",
            "| `GROUPES_BO_COMPARABLES.md` | Synthèse markdown de la simulation BO par BO |",
            "| `01_groupes_bo_scatter.png` | Positionnement des groupes sur score actuel vs volume |",
            "| `02_scores_par_groupe.png` | Score actuel par BO au sein de chaque groupe |",
            "| `03_heatmap_groupes.png` | Profil moyen des groupes |",
            "| `04_analyse_detaillee.png` | Croisement score / répondants / PDTS / volumes |",
            "| `05_systeme_scoring.png` | Barème du challenge |",
            "| `06_simulation_challenge.png` | Visualisation de la projection de challenge |",
            "",
        ]
    )
    return "\n".join(lines)


def export_visuals(frame: pd.DataFrame, group_names: dict[int, str]) -> None:
    sns.set_theme(style="whitegrid", font_scale=1.05)
    colors_4 = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]

    fig, ax = plt.subplots(figsize=(16, 10))
    for group_id in sorted(frame["Groupe"].unique()):
        sub = frame[frame["Groupe"] == group_id]
        color = colors_4[group_id - 1]
        ax.scatter(
            sub["Enquêtes / 2 mois"],
            sub["Nouveau Score Satisfaction"],
            s=sub["est_PDTS"] * 20 + 80,
            alpha=0.8,
            color=color,
            edgecolors="white",
            linewidth=1.5,
            label=group_names[group_id],
            zorder=3,
        )
        for _, row in sub.iterrows():
            ax.annotate(
                row["Base opérationnelle"].replace("BO ", ""),
                (row["Enquêtes / 2 mois"], row["Nouveau Score Satisfaction"]),
                fontsize=8,
                ha="center",
                va="bottom",
                xytext=(0, 8),
                textcoords="offset points",
                fontweight="bold",
            )

    ax.set_xlabel("Enquêtes historiques estimées sur 2 mois", fontsize=13, fontweight="bold")
    ax.set_ylabel("Score de satisfaction actuel (%)", fontsize=13, fontweight="bold")
    ax.set_title(
        "4 groupes de BOs comparables\n(taille des points = PDTS annuel observé)",
        fontsize=15,
        fontweight="bold",
    )
    ax.legend(loc="lower left", fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(ROOT / "01_groupes_bo_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()

    n_groups = len(frame["Groupe"].unique())
    fig, axes = plt.subplots(1, n_groups, figsize=(5 * n_groups, 7), sharey=True)
    if not hasattr(axes, "__len__"):
        axes = [axes]

    for idx, group_id in enumerate(sorted(frame["Groupe"].unique())):
        sub = frame[frame["Groupe"] == group_id].sort_values("Nouveau Score Satisfaction", ascending=True)
        ax = axes[idx]
        color = colors_4[group_id - 1]
        y_pos = range(len(sub))
        bars = ax.barh(
            y_pos,
            sub["Nouveau Score Satisfaction"],
            color=color,
            alpha=0.8,
            edgecolor="white",
        )
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(
            [bo.replace("BO ", "") for bo in sub["Base opérationnelle"]],
            fontsize=9,
        )
        ax.set_xlabel("Score satisfaction actuel (%)", fontsize=10)
        ax.set_title(group_names[group_id], fontsize=11, fontweight="bold", pad=10)
        ax.set_xlim(65, 100)
        for bar, (_, row) in zip(bars, sub.iterrows()):
            ax.text(
                bar.get_width() + 0.3,
                bar.get_y() + bar.get_height() / 2,
                f"{row['Nouveau Score Satisfaction']:.1f}%",
                va="center",
                fontsize=8,
            )

    plt.suptitle("Score actuel par groupe", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(ROOT / "02_scores_par_groupe.png", dpi=150, bbox_inches="tight")
    plt.close()

    metrics = [
        "Nouveau Score Satisfaction",
        "Taux répondants cumulé",
        "Taux PDTS actuel",
        "Enquêtes / 2 mois",
        "Répondants / 2 mois",
        "PDTS / 2 mois",
    ]
    group_summary = frame.groupby("Groupe")[metrics].mean()
    group_summary.index = [group_names[group_id] for group_id in group_summary.index]

    fig, ax = plt.subplots(figsize=(15, 5))
    normalized = group_summary.copy()
    for col in normalized.columns:
        col_min, col_max = normalized[col].min(), normalized[col].max()
        normalized[col] = 0.5 if col_max == col_min else (normalized[col] - col_min) / (col_max - col_min)

    display_labels = [
        "Score\nactuel",
        "Tx répondants\nactuel",
        "Tx PDTS\nactuel",
        "Enquêtes\n/2 mois",
        "Répondants\n/2 mois",
        "PDTS\n/2 mois",
    ]
    sns.heatmap(
        normalized,
        annot=group_summary.round(1).values,
        fmt="",
        cmap="YlOrRd_r",
        ax=ax,
        linewidths=2,
        linecolor="white",
        xticklabels=display_labels,
        cbar_kws={"label": "Performance relative"},
    )
    ax.set_title("Profil moyen des groupes", fontsize=14, fontweight="bold")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(ROOT / "03_heatmap_groupes.png", dpi=150, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    ax1, ax2 = axes[0, 0], axes[0, 1]
    ax3, ax4 = axes[1, 0], axes[1, 1]

    for group_id in sorted(frame["Groupe"].unique()):
        sub = frame[frame["Groupe"] == group_id]
        color = colors_4[group_id - 1]
        ax1.scatter(
            sub["Taux répondants cumulé"],
            sub["Nouveau Score Satisfaction"],
            s=120,
            color=color,
            alpha=0.8,
            label=group_names[group_id],
            edgecolors="white",
            linewidth=1,
        )
        ax2.scatter(
            sub["Taux PDTS actuel"],
            sub["Nouveau Score Satisfaction"],
            s=120,
            color=color,
            alpha=0.8,
            label=group_names[group_id],
            edgecolors="white",
            linewidth=1,
        )
        for _, row in sub.iterrows():
            short_name = row["Base opérationnelle"].replace("BO ", "")
            ax1.annotate(
                short_name,
                (row["Taux répondants cumulé"], row["Nouveau Score Satisfaction"]),
                fontsize=7,
                ha="center",
                va="bottom",
                xytext=(0, 5),
                textcoords="offset points",
            )
            ax2.annotate(
                short_name,
                (row["Taux PDTS actuel"], row["Nouveau Score Satisfaction"]),
                fontsize=7,
                ha="center",
                va="bottom",
                xytext=(0, 5),
                textcoords="offset points",
            )

    ax1.set_xlabel("Taux de répondants actuel (%)")
    ax1.set_ylabel("Score satisfaction actuel (%)")
    ax1.set_title("Score actuel vs taux de répondants")
    ax1.legend(fontsize=8)

    ax2.set_xlabel("Taux PDTS actuel (%)")
    ax2.set_ylabel("Score satisfaction actuel (%)")
    ax2.set_title("Score actuel vs taux PDTS")
    ax2.legend(fontsize=8)

    sorted_df = frame.sort_values("Groupe")
    colors_bars = [colors_4[group_id - 1] for group_id in sorted_df["Groupe"]]
    ax3.bar(
        range(len(sorted_df)),
        sorted_df["Volume challenge / 2 mois"],
        color=colors_bars,
        alpha=0.8,
        edgecolor="white",
    )
    ax3.set_xticks(range(len(sorted_df)))
    ax3.set_xticklabels(
        [bo.replace("BO ", "") for bo in sorted_df["Base opérationnelle"]],
        rotation=90,
        fontsize=7,
    )
    ax3.set_ylabel("Enquêtes simulées sur 2 mois")
    ax3.set_title("Volume challenge simulé par BO")
    handles = [
        mpatches.Patch(color=colors_4[group_id - 1], label=group_names[group_id])
        for group_id in sorted(frame["Groupe"].unique())
    ]
    ax3.legend(handles=handles, fontsize=7, loc="upper right")

    group_data = [
        frame[frame["Groupe"] == group_id]["Score challenge simulé"].values
        for group_id in sorted(frame["Groupe"].unique())
    ]
    bp = ax4.boxplot(
        group_data,
        patch_artist=True,
        tick_labels=[f"Groupe {chr(64 + group_id)}" for group_id in sorted(frame["Groupe"].unique())],
    )
    for patch, group_id in zip(bp["boxes"], sorted(frame["Groupe"].unique())):
        patch.set_facecolor(colors_4[group_id - 1])
        patch.set_alpha(0.7)
    ax4.set_ylabel("Score challenge simulé")
    ax4.set_title("Distribution des scores challenge par groupe")

    plt.suptitle("Analyse détaillée de la simulation", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(ROOT / "04_analyse_detaillee.png", dpi=150, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis("off")
    scoring_text = """
SYSTÈME DE SCORE DU CHALLENGE — 100 points max par BO

1. PROGRESSION DU SCORE SATISFACTION ....................... 40 points
   Points = (gain de score / meilleur gain du groupe) x 40

2. RÉDUCTION DU TAUX DE PDTS ................................ 30 points
   Points = (baisse du taux PDTS / meilleure baisse du groupe) x 30

3. PROGRESSION DU TAUX DE RÉPONDANTS ........................ 20 points
   Points = (gain de taux répondants / meilleur gain du groupe) x 20

4. BONUS EXCELLENCE ........................................ 10 points
   +5 si score fin >= 90 %
   +5 si taux répondants fin >= 18 %

Le classement est calculé au sein de chaque groupe de BO comparables.
"""
    ax.text(
        0.05,
        0.95,
        scoring_text,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(
            boxstyle="round",
            facecolor="#f0f4ff",
            alpha=0.9,
            edgecolor="#2196F3",
            linewidth=2,
        ),
    )
    ax.set_title("Système de score du challenge", fontsize=16, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(ROOT / "05_systeme_scoring.png", dpi=150, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(16, 10))
    for group_id in sorted(frame["Groupe"].unique()):
        sub = frame[frame["Groupe"] == group_id]
        color = colors_4[group_id - 1]
        ax.scatter(
            sub["Taux répondants simulé"],
            sub["Score satisfaction simulé"],
            s=sub["Score challenge simulé"] * 7 + 40,
            color=color,
            alpha=0.82,
            edgecolors="white",
            linewidth=1.2,
            label=group_names[group_id],
        )
        for _, row in sub.iterrows():
            ax.annotate(
                row["Base opérationnelle"].replace("BO ", ""),
                (row["Taux répondants simulé"], row["Score satisfaction simulé"]),
                fontsize=8,
                ha="center",
                va="bottom",
                xytext=(0, 7),
                textcoords="offset points",
            )
    ax.set_xlabel("Taux de répondants simulé (%)")
    ax.set_ylabel("Score satisfaction simulé (%)")
    ax.set_title(
        "Projection de fin de challenge\n(taille des points = score challenge simulé)",
        fontsize=15,
        fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(ROOT / "06_simulation_challenge.png", dpi=150, bbox_inches="tight")
    plt.close()


def export_outputs(frame: pd.DataFrame) -> None:
    simulation_assumptions = {
        1: "Hausse de volume faible (+2 % à +4 %), gain de taux de réponse contenu, PDTS presque incompressibles car déjà bas.",
        2: "Hausse de volume modérée (+4 % à +6 %) et activation relationnelle sur des BO déjà solides pour gagner surtout en répondants.",
        3: "Volumes stables à légèrement haussiers, gains de score surtout liés à la baisse des PDTS sur des volumes importants.",
        4: "Activation plus offensive sur la relation client : meilleur gain de taux de réponse et baisse PDTS plus forte.",
    }

    group_names = {
        group_id: frame.loc[frame["Groupe"] == group_id, "Nom Groupe"].iloc[0]
        for group_id in sorted(frame["Groupe"].unique())
    }
    export_visuals(frame, group_names)

    csv_columns = [
        "Base opérationnelle",
        "Groupe",
        "Nom Groupe",
        "Nouveau Score Satisfaction",
        "Taux répondants cumulé",
        "Taux PDTS actuel",
        "Enquêtes / 2 mois",
        "Répondants / 2 mois",
        "PDTS / 2 mois",
        "Volume challenge / 2 mois",
        "Taux répondants simulé",
        "Répondants simulés / 2 mois",
        "Taux PDTS simulé",
        "PDTS simulés / 2 mois",
        "Score satisfaction simulé",
        "Gain score simulé",
        "Gain taux répondants",
        "Baisse taux PDTS",
        "Points Score",
        "Points PDTS",
        "Points Répondants",
        "Bonus Excellence",
        "Score challenge simulé",
        "Classement groupe",
        "Nouveau Poids Global",
    ]
    csv_export = frame[csv_columns].copy()
    rounded_columns = [
        "Taux répondants cumulé",
        "Taux PDTS actuel",
        "PDTS / 2 mois",
        "Taux répondants simulé",
        "Taux PDTS simulé",
        "Score satisfaction simulé",
        "Gain score simulé",
        "Gain taux répondants",
        "Baisse taux PDTS",
        "Points Score",
        "Points PDTS",
        "Points Répondants",
        "Score challenge simulé",
        "Nouveau Poids Global",
    ]
    csv_export[rounded_columns] = csv_export[rounded_columns].round(2)
    csv_export.sort_values(
        ["Groupe", "Classement groupe", "Score challenge simulé"],
        ascending=[True, True, False],
    ).to_csv(ROOT / "groupes_bo_comparables.csv", index=False, sep=";")

    markdown = render_markdown(frame, simulation_assumptions)
    (ROOT / "GROUPES_BO_COMPARABLES.md").write_text(markdown, encoding="utf-8")


def print_console_summary(frame: pd.DataFrame, method: str, sil_km: float, sil_agg: float) -> None:
    print(f"Silhouette K-Means (k=4): {sil_km:.4f}")
    print(f"Silhouette Hiérarchique (k=4): {sil_agg:.4f}")
    print(f"Méthode retenue: {method}")
    print("\n" + "=" * 88)
    print("SIMULATION DU CHALLENGE SATISFACTION CLIENT — TAUX ET VOLUMES REALISTES")
    print("=" * 88)

    for group_id in sorted(frame["Groupe"].unique()):
        sub = frame[frame["Groupe"] == group_id].sort_values(
            ["Classement groupe", "Score challenge simulé"], ascending=[True, False]
        )
        print(f"\n{'-' * 82}")
        print(f"  Groupe {chr(64 + group_id)} — {len(sub)} BOs")
        print(f"{'-' * 82}")
        print(f"  Volume moyen simulé /2 mois : {sub['Volume challenge / 2 mois'].mean():.1f}")
        print(f"  Tx répondants simulé moyen  : {sub['Taux répondants simulé'].mean():.2f}%")
        print(f"  Tx PDTS simulé moyen        : {sub['Taux PDTS simulé'].mean():.2f}%")
        print(f"  Score simulé moyen          : {sub['Score satisfaction simulé'].mean():.2f}%")
        print()
        print(
            f"    {'BO':<28} {'Vol/2m':>7} {'Tx rép':>7} {'PDTS':>5} "
            f"{'Score':>7} {'Pts':>6} {'Rg':>3}"
        )
        print(f"    {'-' * 72}")
        for _, row in sub.iterrows():
            name = row["Base opérationnelle"].replace("BO ", "")
            print(
                f"    {name:<28} {row['Volume challenge / 2 mois']:>7} "
                f"{row['Taux répondants simulé']:>6.1f}% {row['PDTS simulés / 2 mois']:>5} "
                f"{row['Score satisfaction simulé']:>6.1f}% {row['Score challenge simulé']:>6.1f} "
                f"{row['Classement groupe']:>3}"
            )

    winners = (
        frame.sort_values(["Groupe", "Classement groupe", "Score challenge simulé"], ascending=[True, True, False])
        .groupby("Groupe")
        .head(1)
    )
    print("\n" + "=" * 88)
    print("GAGNANTS SIMULES")
    print("=" * 88)
    for _, row in winners.iterrows():
        print(
            f"  Groupe {chr(64 + row['Groupe'])}: {row['Base opérationnelle']} "
            f"({row['Score challenge simulé']:.1f} pts, score fin {row['Score satisfaction simulé']:.1f}%)"
        )

    print("\nExports mis a jour : GROUPES_BO_COMPARABLES.md, groupes_bo_comparables.csv, graphiques 01 a 06")


df = compute_baseline_metrics(df)
df, method, sil_km, sil_agg = assign_groups(df)
df = build_simulation(df)
df = score_challenge(df)
export_outputs(df)


if __name__ == "__main__":
    print_console_summary(df, method, sil_km, sil_agg)
