# Challenge Satisfaction Client — Mise à jour "Moyenne Mobile > 1 an"

## Journal d'avancement

- ✅ **Source de volume mise à jour** avec le fichier :  
  `/home/ubuntu/.cursor/projects/workspace/uploads/Nombre_d_enqu_tes_envoy_es_en_moyenne_par_jour_par_BO.csv`
- ✅ **Regroupement BO recalculé** (4 groupes, K-Means retenu)
- ✅ **Simulation relancée** avec les nouveaux groupes/volumes
- ✅ **Exports régénérés** (`groupes_bo_comparables.csv`, `simulation_resultats.csv`, PNG 01→09)

---

## Méthode de recalcul des volumes

Pour chaque BO :

1. Calcul de la moyenne des valeurs de **`Nombre d'enquêtes envoyées MM`** sur tout l'historique du fichier.
2. Annualisation : `volume annuel = moyenne_journalière × 365` (arrondi).
3. Conversion période challenge : `Enquêtes / 2 mois = volume annuel / 6` (arrondi).

> Cette logique est maintenant intégrée directement dans `analyse_groupes_bo.py` (avec fallback sur les anciennes valeurs si le CSV est absent).

---

## Résultat du clustering (nouvelle itération)

- **Silhouette K-Means (k=4)** : `0.2583`
- **Silhouette Hiérarchique (k=4)** : `0.2120`
- **Méthode retenue** : **K-Means**

### Composition des groupes

#### Groupe A (3 BOs)
- BO Oyonnax
- BO Privas-Le Cheylard
- BO Valréas

#### Groupe B (8 BOs)
- BO Aubenas-Joyeuse
- BO Roussillon
- BO Ambérieu - Belley
- BO Annonay
- BO Crest Die
- BO Saint Vallier
- BO Valence
- BO Décines

#### Groupe C (8 BOs)
- BO Roannais
- BO Forez
- BO Montélimar
- BO Bourg - Montrevel
- BO l'Arbresle
- BO Heyrieux
- BO Romans
- BO Gleize

#### Groupe D (6 BOs)
- BO Rillieux
- BO Saint Étienne
- BO Vénissieux
- BO Firminy - Saint Bonnet
- BO Givors
- BO Oullins

---

## Simulation relancée — résultats clés

### Gagnants par groupe

| Groupe | Gagnant | Score final /répondant |
|---|---|---:|
| A | **BO Oyonnax** | **+4.59** |
| B | **BO Aubenas-Joyeuse** | **+4.22** |
| C | **BO Forez** | **+3.83** |
| D | **BO Rillieux** | **+3.40** |

### Prix Effort

| Distinction | BO | Progression |
|---|---|---:|
| 🏅 Prix Effort | **BO Firminy - Saint Bonnet** | **+0.32 /répondant** |

Top 3 progressions :
1. BO Firminy - Saint Bonnet (+0.32)
2. BO Heyrieux (+0.28)
3. BO Vénissieux (+0.28)

### Impact global (simulation)

| Indicateur | Avant | Après | Évolution |
|---|---:|---:|---:|
| Score moyen /répondant | +3.43 | +3.62 | +0.18 |
| Total PDTS /2 mois | 44 | 25 | -19 |
| Total répondants /2 mois | 2 912 | 3 415 | +503 |

---

## Fichiers de référence (source de vérité)

- `groupes_bo_comparables.csv` : groupes recalculés + volumes mis à jour
- `simulation_resultats.csv` : résultats complets avant/après
- `analyse_groupes_bo.py` : calcul des groupes, lecture moyenne mobile
- `simulation_challenge.py` : simulation synchronisée automatiquement sur le CSV des groupes

---

## Notes d'exploitation

- Pour relancer la chaîne complète :
  1. `python3 analyse_groupes_bo.py`
  2. `python3 simulation_challenge.py`
- Si le CSV de moyenne mobile n'est pas présent, le script garde les volumes historiques (fallback).

---

## Intégration Power BI — Stats réelles du 1er janvier au 31 mars 2026

Le fichier suivant est exploité :

- `/home/ubuntu/.cursor/projects/workspace/uploads/stats_janvier_mars_2026.csv`

Un script dédié permet de rejouer l'analyse avec la logique de score Power BI :

- `python3 analyse_challenge_reel_jan_mars_2026.py`

### Règle de score utilisée

Le script reprend ta logique :

- `Score Final Challenge = Total Points Gagnés / Total Points Potentiels * 100`
- Matrice de points par `Note x Catégorie` (Particuliers / Professionnels / Entreprises)

> Note: la ligne `1;Particuliers;-4` était en doublon dans le texte source.  
> Le script dédoublonne automatiquement cette ligne sans impact (même valeur).

### Classement réel observé (janv-mars 2026)

Gagnants par groupe :

| Groupe | Gagnant réel | Score Final Challenge |
|---|---|---:|
| A | **BO Valréas** | **8.64** |
| B | **BO Annonay** | **11.53** |
| C | **BO Roannais** | **8.95** |
| D | **BO Rillieux** | **7.94** |

Top 5 global :

1. BO Annonay (11.53)
2. BO Roannais (8.95)
3. BO Valréas (8.64)
4. BO Rillieux (7.94)
5. BO Ambérieu - Belley (7.71)

### Exports générés

- `challenge_reel_jan_mars_2026.csv` (classement détaillé BO, global + groupe)
- `challenge_reel_resume_groupes_2026.csv` (synthèse par groupe)
- `10_challenge_reel_jan_mars_2026.png` (visualisation classement réel)
