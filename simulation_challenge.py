import os
from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
np.random.seed(42)

GROUPING_MODE = os.getenv("GROUPING_MODE", "kmeans").strip().lower()
if GROUPING_MODE not in {"kmeans", "manuel"}:
    raise ValueError("GROUPING_MODE doit valoir 'kmeans' ou 'manuel'")

group_labels = {1: "Groupe A", 2: "Groupe B", 3: "Groupe C", 4: "Groupe D"}
colors_4 = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]


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


def load_stats() -> pd.DataFrame:
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
        "Nombre d'enquêtes envoyées",
        "Nombre de répondants",
        "Nombre de répondants PART",
        "Nombre de répondants PRO",
        "Nombre de répondants ENT",
        "Total Points Gagnés",
        "Total Points Potentiels",
    ]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    print(f"Fichier stats chargé: {stats_path}")
    return df


def load_groups(mode: str) -> pd.DataFrame:
    mode_path = Path(f"/workspace/groupes_bo_comparables_{mode}.csv")
    default_path = Path("/workspace/groupes_bo_comparables.csv")
    group_path = mode_path if mode_path.exists() else default_path
    if not group_path.exists():
        raise FileNotFoundError("Aucun fichier de groupes trouvé. Lance d'abord analyse_groupes_bo.py")

    g = pd.read_csv(group_path, sep=";")
    required = {"Base opérationnelle", "Groupe", "Nom Groupe"}
    missing = required - set(g.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans {group_path}: {', '.join(sorted(missing))}")

    print(f"Fichier groupes chargé: {group_path}")
    return g[list(required)]


def build_simulation(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Rang_ini"] = (
        out.groupby("Groupe")["Score Final Challenge"].rank(method="min", ascending=False).astype(int)
    )
    out["Nb_groupe"] = out.groupby("Groupe")["Base opérationnelle"].transform("count")
    out["Effort_factor"] = np.where(
        out["Nb_groupe"] > 1,
        (out["Rang_ini"] - 1) / (out["Nb_groupe"] - 1),
        0,
    )

    # Hypothèses challenge mai: plus le groupe/BO est bas, plus la marge de progression est élevée.
    base_gain_score = {1: 0.35, 2: 0.50, 3: 0.70, 4: 0.95}
    base_gain_tx = {1: 0.8, 2: 1.0, 3: 1.2, 4: 1.5}

    out["Delta_score_sim"] = (
        out["Groupe"].map(base_gain_score) + 0.9 * out["Effort_factor"]
    ).round(2)
    out["Delta_tx_rep_sim"] = (
        out["Groupe"].map(base_gain_tx) + 1.2 * out["Effort_factor"]
    ).round(2)

    out["Score_final_sim"] = (out["Score Final Challenge"] + out["Delta_score_sim"]).clip(upper=100)
    out["Tx_rep_fin_sim"] = (out["Taux de répondants (%)"] + out["Delta_tx_rep_sim"]).clip(upper=35)

    scale = np.where(out["Taux de répondants (%)"] > 0, out["Tx_rep_fin_sim"] / out["Taux de répondants (%)"], 1.0)
    out["Rep_PART_fin"] = np.round(out["Nombre de répondants PART"] * scale).astype(int)
    out["Rep_PRO_fin"] = np.round(out["Nombre de répondants PRO"] * scale).astype(int)
    out["Rep_ENT_fin"] = np.round(out["Nombre de répondants ENT"] * scale).astype(int)
    out["Rep_fin"] = out["Rep_PART_fin"] + out["Rep_PRO_fin"] + out["Rep_ENT_fin"]

    out["Points_gagnes_fin"] = np.round(out["Total Points Potentiels"] * out["Score_final_sim"] / 100).astype(int)
    out["Points_potentiels_fin"] = out["Total Points Potentiels"].astype(int)

    out["Rang_fin"] = (
        out.groupby("Groupe")["Score_final_sim"].rank(method="min", ascending=False).astype(int)
    )
    out["Gagnant"] = out["Rang_fin"] == 1
    best_effort_idx = out["Delta_score_sim"].idxmax()
    out["Prix_Effort"] = False
    out.loc[best_effort_idx, "Prix_Effort"] = True

    return out


def export_simulation(sim_df: pd.DataFrame, mode: str) -> None:
    export_cols = [
        "Base opérationnelle",
        "Groupe",
        "Nom Groupe",
        "Nombre d'enquêtes envoyées",
        "Nombre de répondants",
        "Rep_fin",
        "Taux de répondants (%)",
        "Tx_rep_fin_sim",
        "Total Points Gagnés",
        "Points_gagnes_fin",
        "Total Points Potentiels",
        "Points_potentiels_fin",
        "Score Final Challenge",
        "Score_final_sim",
        "Delta_score_sim",
        "Rang_ini",
        "Rang_fin",
        "Gagnant",
        "Prix_Effort",
    ]
    out = sim_df[export_cols].sort_values(["Groupe", "Score_final_sim"], ascending=[True, False])
    out.to_csv(f"/workspace/simulation_resultats_{mode}.csv", sep=";", index=False)
    out.to_csv("/workspace/simulation_resultats.csv", sep=";", index=False)


def build_visuals(sim_df: pd.DataFrame, mode: str) -> None:
    sns.set_theme(style="whitegrid", font_scale=1.0)

    # --- Fig 6 ---
    fig, ax = plt.subplots(figsize=(20, 13))
    sim_sorted = sim_df.sort_values(["Groupe", "Score_final_sim"], ascending=[True, False])
    h = 0.35
    for i, (_, r) in enumerate(sim_sorted.iterrows()):
        color = colors_4[r["Groupe"] - 1]
        ax.barh(i + h / 2, r["Score Final Challenge"], height=h, color=color, alpha=0.35, edgecolor=color)
        ax.barh(i - h / 2, r["Score_final_sim"], height=h, color=color, alpha=0.90, edgecolor="white")
        ax.text(
            max(r["Score Final Challenge"], r["Score_final_sim"]) + 0.08,
            i,
            f"{r['Score Final Challenge']:.2f} → {r['Score_final_sim']:.2f}",
            va="center",
            fontsize=7.5,
            fontweight="bold",
        )
        if r["Gagnant"]:
            ax.text(0.05, i, "★", va="center", fontsize=14, color="#FFD600", fontweight="bold")
        if r["Prix_Effort"]:
            ax.text(0.65, i, "PRIX EFFORT", va="center", fontsize=7, fontweight="bold", color="white",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="#FF6F00", alpha=0.9))

    current_g = None
    for i, (_, r) in enumerate(sim_sorted.iterrows()):
        if current_g is not None and r["Groupe"] != current_g:
            ax.axhline(y=i - 0.5, color="gray", linewidth=1.4, linestyle="--", alpha=0.5)
        current_g = r["Groupe"]

    ax.set_yticks(range(len(sim_sorted)))
    ax.set_yticklabels([bo.replace("BO ", "") for bo in sim_sorted["Base opérationnelle"]], fontsize=9)
    ax.set_xlabel("Score Final Challenge (%)")
    ax.set_title(f"Simulation challenge mai — avant/après score final ({mode.upper()})", fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    handles = [mpatches.Patch(color=colors_4[g - 1], alpha=0.9, label=group_labels[g]) for g in sorted(sim_df["Groupe"].unique())]
    handles.append(mpatches.Patch(color="gray", alpha=0.35, label="Avant"))
    handles.append(mpatches.Patch(color="gray", alpha=0.90, label="Après"))
    ax.legend(handles=handles, loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig("/workspace/06_simulation_avant_apres.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Fig 7 ---
    fig, axes = plt.subplots(2, 2, figsize=(22, 18))
    for idx, g in enumerate(sorted(sim_df["Groupe"].unique())):
        ax = axes[idx // 2][idx % 2]
        grp = sim_df[sim_df["Groupe"] == g].sort_values("Score_final_sim")
        y = range(len(grp))
        ax.barh([i + 0.2 for i in y], grp["Nombre de répondants"], height=0.35, alpha=0.35, color=colors_4[g - 1], label="Avant")
        ax.barh([i - 0.2 for i in y], grp["Rep_fin"], height=0.35, alpha=0.90, color=colors_4[g - 1], label="Après")
        ax.set_yticks(list(y))
        ax.set_yticklabels([bo.replace("BO ", "") for bo in grp["Base opérationnelle"]], fontsize=9)
        ax.set_xlabel("Nombre de répondants")
        winner = grp[grp["Gagnant"]].iloc[0]["Base opérationnelle"].replace("BO ", "")
        ax.set_title(f"{group_labels[g]} — gagnant simulé: {winner}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="lower right")
    plt.suptitle(f"Simulation mai — évolution des répondants par groupe ({mode.upper()})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("/workspace/07_decomposition_scoring.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Fig 8 ---
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    ax = axes[0]
    ranked = sim_df.sort_values("Score_final_sim")
    colors_c = [colors_4[g - 1] for g in ranked["Groupe"]]
    ax.barh(range(len(ranked)), ranked["Score_final_sim"], color=colors_c, alpha=0.9, edgecolor="white")
    for i, (_, r) in enumerate(ranked.iterrows()):
        flag = " ★" if r["Gagnant"] else ""
        ax.text(r["Score_final_sim"] + 0.05, i, f"{r['Score_final_sim']:.2f}{flag}", va="center", fontsize=8, fontweight="bold")
    ax.set_yticks(range(len(ranked)))
    ax.set_yticklabels([bo.replace("BO ", "") for bo in ranked["Base opérationnelle"]], fontsize=8)
    ax.set_xlabel("Score final simulé (%)")
    ax.set_title("Classement final simulé")
    handles = [mpatches.Patch(color=colors_4[g - 1], label=group_labels[g]) for g in sorted(sim_df["Groupe"].unique())]
    ax.legend(handles=handles, fontsize=9, loc="lower right")

    ax = axes[1]
    effort = sim_df.sort_values("Delta_score_sim")
    colors_e = [colors_4[g - 1] for g in effort["Groupe"]]
    ax.barh(range(len(effort)), effort["Delta_score_sim"], color=colors_e, alpha=0.9, edgecolor="white")
    for i, (_, r) in enumerate(effort.iterrows()):
        if r["Prix_Effort"]:
            ax.text(r["Delta_score_sim"] + 0.02, i, f"+{r['Delta_score_sim']:.2f} PRIX EFFORT", va="center", fontsize=8, fontweight="bold", color="#E65100")
        else:
            ax.text(r["Delta_score_sim"] + 0.02, i, f"+{r['Delta_score_sim']:.2f}", va="center", fontsize=8, fontweight="bold")
    ax.set_yticks(range(len(effort)))
    ax.set_yticklabels([bo.replace("BO ", "") for bo in effort["Base opérationnelle"]], fontsize=8)
    ax.set_xlabel("Progression simulée du score (%)")
    ax.set_title("Classement progression (prix effort)")
    ax.legend(handles=handles, fontsize=9, loc="lower right")
    plt.suptitle(f"Simulation challenge mai — vue d'ensemble ({mode.upper()})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("/workspace/08_vue_ensemble_simulation.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Fig 9 ---
    for g in sorted(sim_df["Groupe"].unique()):
        grp = sim_df[sim_df["Groupe"] == g].sort_values("Score_final_sim", ascending=False)
        letter = chr(64 + g)
        fig, ax = plt.subplots(figsize=(20, max(3, len(grp) * 0.9 + 2.5)))
        ax.axis("off")
        cols = [
            "BO",
            "Enq",
            "Rép\nAvant",
            "Rép\nAprès",
            "Tx Rép\nAvant",
            "Tx Rép\nAprès",
            "Score\nAvant",
            "Score\nAprès",
            "Δ Score",
            "Rang",
        ]
        table_data = []
        cell_colors = []
        for _, r in grp.iterrows():
            name = r["Base opérationnelle"].replace("BO ", "")
            rang = "★ 1er" if r["Gagnant"] else f"{r['Rang_fin']}e"
            if r["Prix_Effort"]:
                rang += " EFFORT"
            table_data.append(
                [
                    name,
                    str(int(r["Nombre d'enquêtes envoyées"])),
                    str(int(r["Nombre de répondants"])),
                    str(int(r["Rep_fin"])),
                    f"{r['Taux de répondants (%)']:.2f}%",
                    f"{r['Tx_rep_fin_sim']:.2f}%",
                    f"{r['Score Final Challenge']:.2f}",
                    f"{r['Score_final_sim']:.2f}",
                    f"+{r['Delta_score_sim']:.2f}",
                    rang,
                ]
            )
            if r["Gagnant"]:
                row_colors = ["#FFF9C4"] * len(cols)
            elif r["Prix_Effort"]:
                row_colors = ["#FFE0B2"] * len(cols)
            else:
                row_colors = ["#FAFAFA"] * len(cols)
            cell_colors.append(row_colors)

        table = ax.table(cellText=table_data, colLabels=cols, cellLoc="center", loc="center", cellColours=cell_colors)
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.6)
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor(colors_4[g - 1])
                cell.set_text_props(color="white", fontweight="bold", fontsize=8)
                cell.set_edgecolor("white")
            else:
                cell.set_edgecolor("#E0E0E0")
        winner = grp.iloc[0]["Base opérationnelle"].replace("BO ", "")
        ax.set_title(f"{group_labels[g]} — gagnant simulé: {winner}", fontsize=11, fontweight="bold", color=colors_4[g - 1], pad=20)
        plt.tight_layout()
        plt.savefig(f"/workspace/09_{letter}_tableau_resultats.png", dpi=150, bbox_inches="tight")
        plt.close()


stats_df = load_stats()
groups_df = load_groups(GROUPING_MODE)
df = stats_df.merge(groups_df, on="Base opérationnelle", how="left")
if df["Groupe"].isna().any():
    missing = ", ".join(sorted(df[df["Groupe"].isna()]["Base opérationnelle"].tolist()))
    raise ValueError(f"BO sans groupe détectée: {missing}")
df["Groupe"] = df["Groupe"].astype(int)

sim = build_simulation(df)
export_simulation(sim, GROUPING_MODE)
build_visuals(sim, GROUPING_MODE)

print("\n" + "=" * 95)
print(f"SIMULATION CHALLENGE MAI — MODE {GROUPING_MODE.upper()}")
print("=" * 95)
for g in sorted(sim["Groupe"].unique()):
    grp = sim[sim["Groupe"] == g].sort_values("Score_final_sim", ascending=False)
    winner = grp.iloc[0]["Base opérationnelle"]
    print(
        f"{group_labels[g]} ({len(grp)} BOs) — gagnant simulé: {winner} | "
        f"moy score {grp['Score Final Challenge'].mean():.2f} → {grp['Score_final_sim'].mean():.2f}"
    )

effort_winner = sim[sim["Prix_Effort"]].iloc[0]
print(
    f"\nPRIX EFFORT: {effort_winner['Base opérationnelle']} "
    f"(+{effort_winner['Delta_score_sim']:.2f} pts de score)"
)
print("\nExports:")
print(" - /workspace/simulation_resultats.csv (mode actif)")
print(f" - /workspace/simulation_resultats_{GROUPING_MODE}.csv")
print("Graphiques exportés: 06, 07, 08, 09_A/B/C/D")
