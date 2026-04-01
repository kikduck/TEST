# Challenge Satisfaction Client — Regroupement Janv-Mars 2026 (Clean)

## Objectif

Refaire les groupes BO **à partir des stats réelles du 1er janvier au 31 mars 2026**  
et relancer une simulation de challenge pour une campagne cible en mai.

Fichier source:
- `/home/ubuntu/.cursor/projects/workspace/uploads/stats_janvier_mars_2026.csv`

---

## Scoring utilisé (Power BI)

- `Score Final Challenge = Total Points Gagnés / Total Points Potentiels * 100`
- Matrice de points (Note x Catégorie):
  - 4/4: Part +2 | Pro +40 | Ent +140
  - 3/4: Part +1 | Pro +30 | Ent +100
  - 2/4: Part -1 | Pro -17 | Ent -73
  - 1/4: Part -4 | Pro -70 | Ent -292

---

## Deux modes de regroupement disponibles

Le script `analyse_groupes_bo.py` gère maintenant **2 stratégies** :

1. **Mode KMeans** (`GROUPING_MODE=kmeans`, mode par défaut)
   - Clustering multi-critères sur:
     - Score Final Challenge
     - Taux de répondants
     - Taux SAT NETTE Composite
     - Volumes (enquêtes, répondants)
   - Relabeling automatique des groupes A→D par score moyen décroissant.

2. **Mode Manuel** (`GROUPING_MODE=manuel`)
   - Règle explicite basée sur les quartiles du `Score Final Challenge`.
   - Groupes lisibles et rejouables sans ML.

---

## Résultats actuels (mode par défaut = KMeans)

- Silhouette KMeans: **0.2993**
- Silhouette Ward: **0.2789**
- Méthode retenue: **K-Means**

### Gagnants simulation challenge mai (KMeans)

| Groupe | Gagnant simulé |
|---|---|
| A | **BO Annonay** |
| B | **BO Firminy - Saint Bonnet** |
| C | **BO Saint Vallier** |
| D | **BO l'Arbresle** |

**Prix Effort simulé**: **BO l'Arbresle** (+1.85 pts)

---

## Exécution rapide

### KMeans (défaut)
1. `python3 analyse_groupes_bo.py`
2. `python3 simulation_challenge.py`

### Manuel
1. `GROUPING_MODE=manuel python3 analyse_groupes_bo.py`
2. `GROUPING_MODE=manuel python3 simulation_challenge.py`

---

## Fichiers générés

### Groupes
- `groupes_bo_comparables.csv` (mode actif)
- `groupes_bo_comparables_kmeans.csv`
- `groupes_bo_comparables_manuel.csv`
- Graphiques `01` à `05`

### Simulation
- `simulation_resultats.csv` (mode actif)
- `simulation_resultats_kmeans.csv`
- `simulation_resultats_manuel.csv`
- Graphiques `06`, `07`, `08`, `09_A/B/C/D`

### Données réelles de cadrage
- `challenge_reel_jan_mars_2026.csv`
- `challenge_reel_resume_groupes_2026.csv`
- `10_challenge_reel_jan_mars_2026.png`
