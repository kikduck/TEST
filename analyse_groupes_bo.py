import os
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

N_GROUPS = 4
GROUPING_MODE = os.getenv("GROUPING_MODE", "kmeans").strip().lower()
if GROUPING_MODE not in {"kmeans", "manuel"}:
    raise ValueError("GROUPING_MODE doit valoir 'kmeans' ou 'manuel'")


def parse_fr_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str)
        .str.replace("\u202f", "", regex=False)
        .str.replace("\xa0", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace(",", ".", regex=False)
        .str.strip(),
        errors="coerce",
    )


def load_stats_jan_mars() -> pd.DataFrame:
    candidates = [
        Path("/workspace/stats_janvier_mars_2026.csv"),
        Path("/workspace/data/stats_janvier_mars_2026.csv"),
        Path("/home/ubuntu/.cursor/projects/workspace/uploads/stats_janvier_mars_2026.csv"),
    ]
    stats_path = next((p for p in candidates if p.exists()), None)
    if stats_path is None:
        raise FileNotFoundError("stats_janvier_mars_2026.csv introuvable")

    df = pd.read_csv(stats_path)
    df["Score Final Challenge"] = parse_fr_number(df["Score Final Challenge"])
    df["Taux de répondants (%)"] = parse_fr_number(df["Taux de répondants"])
    df["Taux SAT NETTE Composite (%)"] = parse_fr_number(df["Taux SAT NETTE Composite"])

    numeric_cols = [
        "Nombre de répondants ENT",
        "Nombre de répondants PART",
        "Nombre de répondants PRO",
        "Nombre de répondants",
        "Nombre d'enquêtes envoyées",
        "Total Points Gagnés",
        "Total Points Potentiels",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["Score recalculé (%)"] = np.where(
        df["Total Points Potentiels"] > 0,
        df["Total Points Gagnés"] / df["Total Points Potentiels"] * 100,
        0,
    )
    df["Écart score (pts)"] = df["Score Final Challenge"] - df["Score recalculé (%)"]
    print(f"Fichier source chargé: {stats_path}")
    return df


def assign_groups_kmeans(df: pd.DataFrame) -> tuple[pd.Series, str]:
    features = [
        "Score Final Challenge",
        "Taux de répondants (%)",
        "Taux SAT NETTE Composite (%)",
        "Nombre d'enquêtes envoyées",
        "Nombre de répondants",
    ]
    x = df[features].values
    x_scaled = StandardScaler().fit_transform(x)

    km = KMeans(n_clusters=N_GROUPS, random_state=42, n_init=40)
    labels_km = km.fit_predict(x_scaled)
    sil_km = silhouette_score(x_scaled, labels_km)

    agg = AgglomerativeClustering(n_clusters=N_GROUPS, linkage="ward")
    labels_agg = agg.fit_predict(x_scaled)
    sil_agg = silhouette_score(x_scaled, labels_agg)

    print(f"[KMEANS] Silhouette K-Means: {sil_km:.4f}")
    print(f"[KMEANS] Silhouette Ward   : {sil_agg:.4f}")
    if sil_agg > sil_km:
        return pd.Series(labels_agg, index=df.index), "Hiérarchique (Ward)"
    return pd.Series(labels_km, index=df.index), "K-Means"


def assign_groups_manual(df: pd.DataFrame) -> tuple[pd.Series, str]:
    # Mode manuel "clean": 4 niveaux équilibrés par quartiles de score réel.
    q25 = df["Score Final Challenge"].quantile(0.25)
    q50 = df["Score Final Challenge"].quantile(0.50)
    q75 = df["Score Final Challenge"].quantile(0.75)

    labels = np.select(
        [
            df["Score Final Challenge"] >= q75,
            (df["Score Final Challenge"] >= q50) & (df["Score Final Challenge"] < q75),
            (df["Score Final Challenge"] >= q25) & (df["Score Final Challenge"] < q50),
        ],
        [0, 1, 2],
        default=3,
    )
    print(f"[MANUEL] Seuils score: Q25={q25:.2f}, Q50={q50:.2f}, Q75={q75:.2f}")
    return pd.Series(labels, index=df.index), "Manuel (quartiles score)"


def postprocess_groups(df: pd.DataFrame, raw_labels: pd.Series, method_name: str, mode: str) -> pd.DataFrame:
    out = df.copy()
    out["Groupe_raw"] = raw_labels.astype(int)
    means = (
        out.groupby("Groupe_raw")["Score Final Challenge"]
        .mean()
        .sort_values(ascending=False)
    )
    mapping = {old: i + 1 for i, old in enumerate(means.index)}
    out["Groupe"] = out["Groupe_raw"].map(mapping).astype(int)

    group_names = {}
    for g in sorted(out["Groupe"].unique()):
        n_bos = int((out["Groupe"] == g).sum())
        group_names[g] = f"Groupe {chr(64 + g)} ({n_bos} BOs)"
    out["Nom Groupe"] = out["Groupe"].map(group_names)
    out["Mode regroupement"] = mode
    out["Méthode regroupement"] = method_name
    out["Rang global réel"] = (
        out["Score Final Challenge"].rank(method="min", ascending=False).astype(int)
    )
    out["Rang groupe réel"] = (
        out.groupby("Groupe")["Score Final Challenge"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    return out


def export_groups(df: pd.DataFrame, suffix: str) -> None:
    cols = [
        "Base opérationnelle",
        "Groupe",
        "Nom Groupe",
        "Rang groupe réel",
        "Rang global réel",
        "Score Final Challenge",
        "Taux SAT NETTE Composite (%)",
        "Taux de répondants (%)",
        "Nombre d'enquêtes envoyées",
        "Nombre de répondants",
        "Nombre de répondants PART",
        "Nombre de répondants PRO",
        "Nombre de répondants ENT",
        "Total Points Gagnés",
        "Total Points Potentiels",
        "Score recalculé (%)",
        "Écart score (pts)",
        "Mode regroupement",
        "Méthode regroupement",
    ]
    out = df[cols].sort_values(["Groupe", "Score Final Challenge"], ascending=[True, False])
    out.to_csv(f"/workspace/groupes_bo_comparables_{suffix}.csv", sep=";", index=False)


def build_visuals(df: pd.DataFrame, mode: str) -> None:
    sns.set_theme(style="whitegrid", font_scale=1.05)
    colors_4 = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]
    group_names = {g: df[df["Groupe"] == g]["Nom Groupe"].iloc[0] for g in sorted(df["Groupe"].unique())}

    # Fig 1
    fig, ax = plt.subplots(figsize=(16, 10))
    for g in sorted(df["Groupe"].unique()):
        sub = df[df["Groupe"] == g]
        color = colors_4[g - 1]
        ax.scatter(
            sub["Nombre d'enquêtes envoyées"],
            sub["Score Final Challenge"],
            s=sub["Nombre de répondants"] * 2 + 60,
            alpha=0.8,
            color=color,
            edgecolors="white",
            linewidth=1.2,
            label=group_names[g],
        )
        for _, row in sub.iterrows():
            ax.annotate(
                row["Base opérationnelle"].replace("BO ", ""),
                (row["Nombre d'enquêtes envoyées"], row["Score Final Challenge"]),
                fontsize=8,
                ha="center",
                va="bottom",
                xytext=(0, 7),
                textcoords="offset points",
            )
    ax.set_xlabel("Enquêtes envoyées (janv-mars 2026)")
    ax.set_ylabel("Score Final Challenge (%)")
    ax.set_title(f"Groupes BO comparables — mode {mode.upper()} (base réelle janv-mars 2026)")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("/workspace/01_groupes_bo_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Fig 2
    n_groups = len(df["Groupe"].unique())
    fig, axes = plt.subplots(1, n_groups, figsize=(5 * n_groups, 7), sharey=True)
    if not hasattr(axes, "__len__"):
        axes = [axes]
    for idx, g in enumerate(sorted(df["Groupe"].unique())):
        sub = df[df["Groupe"] == g].sort_values("Score Final Challenge")
        ax = axes[idx]
        bars = ax.barh(
            range(len(sub)),
            sub["Score Final Challenge"],
            color=colors_4[g - 1],
            alpha=0.85,
            edgecolor="white",
        )
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels([bo.replace("BO ", "") for bo in sub["Base opérationnelle"]], fontsize=9)
        ax.set_xlabel("Score Final Challenge (%)")
        ax.set_title(group_names[g], fontsize=11, fontweight="bold")
        for bar, (_, row) in zip(bars, sub.iterrows()):
            ax.text(bar.get_width() + 0.08, bar.get_y() + bar.get_height() / 2, f"{row['Score Final Challenge']:.2f}", va="center", fontsize=8)
    plt.suptitle(f"Score final challenge par groupe — mode {mode.upper()}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("/workspace/02_scores_par_groupe.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Fig 3
    metrics = [
        "Score Final Challenge",
        "Taux de répondants (%)",
        "Taux SAT NETTE Composite (%)",
        "Nombre d'enquêtes envoyées",
        "Nombre de répondants",
        "Total Points Gagnés",
        "Total Points Potentiels",
    ]
    group_summary = df.groupby("Groupe")[metrics].mean()
    group_summary.index = [group_names[g] for g in group_summary.index]
    normalized = group_summary.copy()
    for c in normalized.columns:
        mn, mx = normalized[c].min(), normalized[c].max()
        normalized[c] = 0.5 if mx == mn else (normalized[c] - mn) / (mx - mn)
    fig, ax = plt.subplots(figsize=(16, 5))
    sns.heatmap(
        normalized,
        annot=group_summary.round(2).values,
        fmt="",
        cmap="YlOrRd_r",
        ax=ax,
        linewidths=1.6,
        linecolor="white",
        cbar_kws={"label": "Performance relative"},
    )
    ax.set_title(f"Profil moyen par groupe — mode {mode.upper()}")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig("/workspace/03_heatmap_groupes.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Fig 4
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    ax1, ax2, ax3, ax4 = axes.flatten()
    for g in sorted(df["Groupe"].unique()):
        sub = df[df["Groupe"] == g]
        color = colors_4[g - 1]
        ax1.scatter(sub["Taux de répondants (%)"], sub["Score Final Challenge"], color=color, alpha=0.8, s=100, label=group_names[g], edgecolors="white")
        ax2.scatter(sub["Taux SAT NETTE Composite (%)"], sub["Score Final Challenge"], color=color, alpha=0.8, s=100, label=group_names[g], edgecolors="white")
    ax1.set_xlabel("Taux de répondants (%)")
    ax1.set_ylabel("Score Final Challenge (%)")
    ax1.set_title("Score vs Taux de répondants")
    ax1.legend(fontsize=8)
    ax2.set_xlabel("Taux SAT NETTE Composite (%)")
    ax2.set_ylabel("Score Final Challenge (%)")
    ax2.set_title("Score vs SAT NETTE")
    ax2.legend(fontsize=8)

    sorted_df = df.sort_values(["Groupe", "Score Final Challenge"], ascending=[True, False])
    colors_bars = [colors_4[g - 1] for g in sorted_df["Groupe"]]
    ax3.bar(range(len(sorted_df)), sorted_df["Nombre d'enquêtes envoyées"], color=colors_bars, alpha=0.85, edgecolor="white")
    ax3.set_xticks(range(len(sorted_df)))
    ax3.set_xticklabels([bo.replace("BO ", "") for bo in sorted_df["Base opérationnelle"]], rotation=90, fontsize=7)
    ax3.set_ylabel("Enquêtes envoyées (janv-mars)")
    ax3.set_title("Volumes d'enquêtes par BO")
    handles = [mpatches.Patch(color=colors_4[g - 1], label=group_names[g]) for g in sorted(df["Groupe"].unique())]
    ax3.legend(handles=handles, fontsize=7, loc="upper right")

    group_data = [df[df["Groupe"] == g]["Score Final Challenge"].values for g in sorted(df["Groupe"].unique())]
    bp = ax4.boxplot(group_data, patch_artist=True, tick_labels=[f"Groupe {chr(64 + g)}" for g in sorted(df["Groupe"].unique())])
    for patch, g in zip(bp["boxes"], sorted(df["Groupe"].unique())):
        patch.set_facecolor(colors_4[g - 1])
        patch.set_alpha(0.7)
    ax4.set_ylabel("Score Final Challenge (%)")
    ax4.set_title("Distribution des scores par groupe")

    plt.suptitle(f"Analyse détaillée des groupes — mode {mode.upper()}", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig("/workspace/04_analyse_detaillee.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Fig 5
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis("off")
    scoring_text = """
SYSTÈME DE SCORE CHALLENGE (Power BI)

Total Points Gagnés      = Σ score(Note, Catégorie)
Total Points Potentiels  = Σ (Enquêtes envoyées × score max catégorie)
Score Final Challenge    = (Points Gagnés / Points Potentiels) × 100

Matrice de points :
  4/4 : Part +2 | Pro +40 | Ent +140
  3/4 : Part +1 | Pro +30 | Ent +100
  2/4 : Part -1 | Pro -17 | Ent -73
  1/4 : Part -4 | Pro -70 | Ent -292

Mode de regroupement actif : {mode}
"""
    ax.text(
        0.05,
        0.95,
        scoring_text.format(mode=mode.upper()),
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="#f0f4ff", alpha=0.95, edgecolor="#2196F3", linewidth=2),
    )
    ax.set_title("Système de score utilisé", fontsize=16, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig("/workspace/05_systeme_scoring.png", dpi=150, bbox_inches="tight")
    plt.close()


base = load_stats_jan_mars()

kmeans_labels, kmeans_method = assign_groups_kmeans(base)
manual_labels, manual_method = assign_groups_manual(base)

df_kmeans = postprocess_groups(base, kmeans_labels, kmeans_method, "kmeans")
df_manual = postprocess_groups(base, manual_labels, manual_method, "manuel")
export_groups(df_kmeans, "kmeans")
export_groups(df_manual, "manuel")

selected = df_kmeans if GROUPING_MODE == "kmeans" else df_manual
selected_export = selected.sort_values(["Groupe", "Score Final Challenge"], ascending=[True, False])
selected_export.to_csv("/workspace/groupes_bo_comparables.csv", sep=";", index=False)

build_visuals(selected, GROUPING_MODE)

print("\n" + "=" * 90)
print(f"GROUPES BO COMPARABLES — BASE JANV-MARS 2026 — MODE {GROUPING_MODE.upper()}")
print("=" * 90)
for g in sorted(selected["Groupe"].unique()):
    sub = selected[selected["Groupe"] == g].sort_values("Score Final Challenge", ascending=False)
    print(f"\n{selected[selected['Groupe'] == g]['Nom Groupe'].iloc[0]}")
    print("-" * 90)
    avg_sent = sub["Nombre d'enquêtes envoyées"].mean()
    print(
        f"Moy. score: {sub['Score Final Challenge'].mean():.2f} | "
        f"Moy. tx répondants: {sub['Taux de répondants (%)'].mean():.2f}% | "
        f"Moy. volumes envoyés: {avg_sent:.0f}"
    )
    for _, r in sub.iterrows():
        name = r["Base opérationnelle"].replace("BO ", "")
        enq = int(r["Nombre d'enquêtes envoyées"])
        print(
            f"  - {name:<28}  score={r['Score Final Challenge']:>5.2f}  "
            f"tx_rep={r['Taux de répondants (%)']:>5.2f}%  "
            f"enq={enq:>4}"
        )

print("\nExports CSV:")
print(" - /workspace/groupes_bo_comparables.csv (mode actif)")
print(" - /workspace/groupes_bo_comparables_kmeans.csv")
print(" - /workspace/groupes_bo_comparables_manuel.csv")
print("Graphiques exportés: 01 à 05")
