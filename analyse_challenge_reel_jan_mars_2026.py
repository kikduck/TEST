from pathlib import Path

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_fr_number(series: pd.Series) -> pd.Series:
    """Convertit des valeurs FR (virgule, %) en float."""
    return pd.to_numeric(
        series.astype(str)
        .str.replace("\u202f", "", regex=False)
        .str.replace("\xa0", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace(",", ".", regex=False)
        .str.strip(),
        errors="coerce",
    )


stats_candidates = [
    Path("/workspace/stats_janvier_mars_2026.csv"),
    Path("/workspace/data/stats_janvier_mars_2026.csv"),
    Path("/home/ubuntu/.cursor/projects/workspace/uploads/stats_janvier_mars_2026.csv"),
]
stats_path = next((p for p in stats_candidates if p.exists()), None)
if stats_path is None:
    raise FileNotFoundError("Impossible de trouver stats_janvier_mars_2026.csv")

groupes_path = Path("/workspace/groupes_bo_comparables.csv")
if not groupes_path.exists():
    raise FileNotFoundError("groupes_bo_comparables.csv introuvable. Lance d'abord analyse_groupes_bo.py")

stats = pd.read_csv(stats_path)

# Matrice de scoring fournie (avec dédoublonnage de la ligne 1/4 Particuliers)
scoring_matrix = pd.DataFrame(
    [
        {"Note": 4, "Catégorie": "Particuliers", "Score": 2},
        {"Note": 4, "Catégorie": "Professionnels", "Score": 40},
        {"Note": 4, "Catégorie": "Entreprises", "Score": 140},
        {"Note": 3, "Catégorie": "Particuliers", "Score": 1},
        {"Note": 3, "Catégorie": "Professionnels", "Score": 30},
        {"Note": 3, "Catégorie": "Entreprises", "Score": 100},
        {"Note": 2, "Catégorie": "Particuliers", "Score": -1},
        {"Note": 2, "Catégorie": "Professionnels", "Score": -17},
        {"Note": 2, "Catégorie": "Entreprises", "Score": -73},
        {"Note": 1, "Catégorie": "Particuliers", "Score": -4},
        {"Note": 1, "Catégorie": "Particuliers", "Score": -4},
        {"Note": 1, "Catégorie": "Professionnels", "Score": -70},
        {"Note": 1, "Catégorie": "Entreprises", "Score": -292},
    ]
).drop_duplicates(subset=["Note", "Catégorie"], keep="first")

stats["Score Final Challenge"] = parse_fr_number(stats["Score Final Challenge"])
stats["Taux de répondants (%)"] = parse_fr_number(stats["Taux de répondants"])
stats["Taux SAT NETTE Composite (%)"] = parse_fr_number(stats["Taux SAT NETTE Composite"])

for col in [
    "Nombre de répondants ENT",
    "Nombre de répondants PART",
    "Nombre de répondants PRO",
    "Nombre de répondants",
    "Nombre d'enquêtes envoyées",
    "Total Points Gagnés",
    "Total Points Potentiels",
]:
    stats[col] = pd.to_numeric(stats[col], errors="coerce").fillna(0)

# Contrôle de cohérence sur le score final
stats["Score recalculé (points gagnés / potentiels)"] = (
    stats["Total Points Gagnés"] / stats["Total Points Potentiels"] * 100
)
stats["Écart score (pts)"] = (
    stats["Score Final Challenge"] - stats["Score recalculé (points gagnés / potentiels)"]
)

groupes = pd.read_csv(groupes_path, sep=";")[["Base opérationnelle", "Groupe", "Nom Groupe"]]
df = stats.merge(groupes, on="Base opérationnelle", how="left")

missing_group = df[df["Groupe"].isna()]["Base opérationnelle"].tolist()
if missing_group:
    raise ValueError("BO sans groupe détectées: " + ", ".join(missing_group))

df["Groupe"] = df["Groupe"].astype(int)
df["Rang Groupe"] = (
    df.groupby("Groupe")["Score Final Challenge"].rank(method="min", ascending=False).astype(int)
)
df["Rang Global"] = df["Score Final Challenge"].rank(method="min", ascending=False).astype(int)
df["Gagnant Groupe"] = df["Rang Groupe"] == 1

leaders = (
    df.sort_values(["Groupe", "Rang Groupe"])
    .groupby("Groupe", as_index=False)
    .first()[["Groupe", "Score Final Challenge"]]
    .rename(columns={"Score Final Challenge": "Leader Score Groupe"})
)
df = df.merge(leaders, on="Groupe", how="left")
df["Écart au leader groupe"] = (df["Leader Score Groupe"] - df["Score Final Challenge"]).round(2)

# Projection simple pour un challenge en mai: objectif +1 point vs score actuel, capé à 100.
df["Objectif score mai (base +1pt)"] = (df["Score Final Challenge"] + 1).clip(upper=100).round(2)

# Export principal
export_cols = [
    "Base opérationnelle",
    "Groupe",
    "Nom Groupe",
    "Rang Groupe",
    "Rang Global",
    "Gagnant Groupe",
    "Score Final Challenge",
    "Objectif score mai (base +1pt)",
    "Écart au leader groupe",
    "Total Points Gagnés",
    "Total Points Potentiels",
    "Nombre d'enquêtes envoyées",
    "Nombre de répondants",
    "Nombre de répondants PART",
    "Nombre de répondants PRO",
    "Nombre de répondants ENT",
    "Taux de répondants (%)",
    "Taux SAT NETTE Composite (%)",
    "Score recalculé (points gagnés / potentiels)",
    "Écart score (pts)",
]

df_out = df[export_cols].sort_values(["Groupe", "Rang Groupe", "Base opérationnelle"])
df_out.to_csv("/workspace/challenge_reel_jan_mars_2026.csv", sep=";", index=False)

# Export résumé groupe
summary = (
    df.groupby(["Groupe", "Nom Groupe"], as_index=False)
    .agg(
        Nb_BO=("Base opérationnelle", "count"),
        Score_moyen=("Score Final Challenge", "mean"),
        Score_max=("Score Final Challenge", "max"),
        Score_min=("Score Final Challenge", "min"),
        Taux_rep_moyen=("Taux de répondants (%)", "mean"),
    )
    .sort_values("Groupe")
)
summary.to_csv("/workspace/challenge_reel_resume_groupes_2026.csv", sep=";", index=False)

# Figure classement
colors = {1: "#2196F3", 2: "#4CAF50", 3: "#FF9800", 4: "#E91E63"}
plot_df = df_out.sort_values("Score Final Challenge", ascending=True)
bar_colors = [colors[g] for g in plot_df["Groupe"]]

fig, ax = plt.subplots(figsize=(16, 11))
bars = ax.barh(
    range(len(plot_df)),
    plot_df["Score Final Challenge"],
    color=bar_colors,
    alpha=0.88,
    edgecolor="white",
)
for i, (_, r) in enumerate(plot_df.iterrows()):
    winner_flag = " ★" if r["Gagnant Groupe"] else ""
    ax.text(
        r["Score Final Challenge"] + 0.1,
        i,
        f"{r['Score Final Challenge']:.2f}{winner_flag}",
        va="center",
        fontsize=8,
        fontweight="bold",
    )

ax.set_yticks(range(len(plot_df)))
ax.set_yticklabels([bo.replace("BO ", "") for bo in plot_df["Base opérationnelle"]], fontsize=8)
ax.set_xlabel("Score Final Challenge (janv-mars 2026, %)")
ax.set_title("Classement réel du challenge (Power BI) — Janvier à Mars 2026", fontsize=13, fontweight="bold")
ax.grid(axis="x", alpha=0.25)

legend_handles = [
    plt.Rectangle((0, 0), 1, 1, color=colors[g], label=f"Groupe {chr(64 + g)}")
    for g in sorted(df["Groupe"].unique())
]
ax.legend(handles=legend_handles, loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig("/workspace/10_challenge_reel_jan_mars_2026.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"Fichier source lu: {stats_path}")
print("CSV exporté: challenge_reel_jan_mars_2026.csv")
print("Résumé exporté: challenge_reel_resume_groupes_2026.csv")
print("Figure exportée: 10_challenge_reel_jan_mars_2026.png")
print("\nTOP 5 global (Score Final Challenge):")
print(
    df.sort_values("Score Final Challenge", ascending=False)[
        ["Base opérationnelle", "Groupe", "Score Final Challenge"]
    ]
    .head(5)
    .to_string(index=False)
)
print("\nGagnants par groupe:")
print(
    df[df["Gagnant Groupe"]]
    .sort_values("Groupe")[["Groupe", "Base opérationnelle", "Score Final Challenge"]]
    .to_string(index=False)
)
print("\nMatrice de scoring utilisée:")
print(scoring_matrix.sort_values(["Catégorie", "Note"], ascending=[True, False]).to_string(index=False))
