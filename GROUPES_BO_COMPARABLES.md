# Challenge Satisfaction Client — Groupes de Bases Opérationnelles Comparables

## Objectif

Constituer des groupes de BOs équitables pour un challenge visant à :
- **Améliorer le Nouveau Score de Satisfaction** (Taux 4/4 + Taux 3/4 - Taux 1/4)
- **Augmenter le taux de répondants** pour obtenir davantage de notes 4/4 et 3/4
- **Réduire le nombre de PDTS** (réponses 1/4)

## Méthodologie de regroupement

Les groupes sont constitués via une analyse de clustering (K-Means) sur 4 critères normalisés :

| Critère | Justification |
|---------|---------------|
| **Nombre d'enquêtes envoyées** | Mesure le volume d'activité — une BO à 1 600 enquêtes n'a pas la même base qu'une BO à 285 |
| **Taux de répondants cumulé** | Capacité actuelle à engager les clients dans l'enquête |
| **Nombre de PDTS (est_PDTS)** | Volume absolu de détracteurs (réponses 1/4) à réduire |
| **Nouveau Score de Satisfaction** | Niveau de performance actuel — les BOs se challengent entre pairs |

Le nombre optimal de groupes (6) a été déterminé par le score silhouette.

> **Cas particulier** : BO Oyonnax forme un groupe à part (0 PDTS, 285 enquêtes seulement, score 95%). Elle pourra être intégrée au challenge en mode « mentor/observateur » ou rattachée au Groupe 2 quand son volume augmentera.

---

## Les 6 Groupes

### Groupe 1 — 🏆 Haute Performance / Petit Volume (1 BO)

> **Profil** : BO en démarrage ou très petit volume, avec un score exceptionnel.

| Base opérationnelle | Score Satisfaction | Taux Composite | PDTS | Enquêtes | Taux Répondants | Poids Global |
|---|---|---|---|---|---|---|
| **BO Oyonnax** | 95,00 % | 0,00 % | 0 | 285 | 14,04 % | 0,13 % |

**Axes de challenge** : Maintenir le score d'excellence tout en augmentant le volume d'enquêtes et le taux de répondants.

---

### Groupe 2 — 📊 Performance Intermédiaire-Haute / Volume Moyen (5 BOs)

> **Profil** : BOs avec un bon score de satisfaction (85-90%), un taux de répondants modéré (~13%) et un volume moyen d'enquêtes. Marge de progression sur l'engagement des répondants.

| Base opérationnelle | Score Satisfaction | Taux Composite | PDTS | Enquêtes | Taux Répondants | Poids Global |
|---|---|---|---|---|---|---|
| **BO Roussillon** | 90,00 % | 6,41 % | 5 | 1 229 | 13,02 % | 0,99 % |
| **BO Ambérieu - Belley** | 88,89 % | 3,24 % | 7 | 1 353 | 13,97 % | 1,25 % |
| **BO Annonay** | 88,07 % | 3,04 % | 5 | 844 | 12,91 % | 0,54 % |
| **BO Valence** | 85,52 % | 4,04 % | 9 | 1 083 | 13,39 % | 0,79 % |
| **BO Décines** | 84,96 % | 3,08 % | 8 | 987 | 13,48 % | 0,69 % |

| Indicateur | Moyenne du groupe |
|---|---|
| Score Satisfaction | **87,49 %** |
| Taux Composite | 3,96 % |
| Taux Répondants | 13,35 % |
| PDTS moyen | 6,8 |
| Enquêtes | 1 099 |

**Axes de challenge** :
- Augmenter le taux de répondants de **13% → 16%** pour maximiser les 4/4 et 3/4
- Maintenir le nombre de PDTS sous contrôle (< 7)
- Objectif score : **dépasser les 90%**

---

### Groupe 3 — 📊 Performance Intermédiaire-Haute / Volume Moyen (5 BOs)

> **Profil** : BOs avec un bon score (83-90%), un taux de répondants déjà élevé (~17%) mais sur des volumes plus petits. Territoire géographiquement plus rural/montagnard.

| Base opérationnelle | Score Satisfaction | Taux Composite | PDTS | Enquêtes | Taux Répondants | Poids Global |
|---|---|---|---|---|---|---|
| **BO Aubenas-Joyeuse** | 90,10 % | 1,68 % | 4 | 1 130 | 16,99 % | 1,30 % |
| **BO Privas-Le Cheylard** | 87,86 % | 9,40 % | 7 | 755 | 18,54 % | 0,78 % |
| **BO Crest Die** | 87,41 % | 2,89 % | 8 | 922 | 15,51 % | 0,80 % |
| **BO Saint Vallier** | 85,61 % | 4,81 % | 8 | 817 | 16,16 % | 0,69 % |
| **BO Valréas** | 83,33 % | 11,77 % | 6 | 571 | 16,81 % | 0,41 % |

| Indicateur | Moyenne du groupe |
|---|---|
| Score Satisfaction | **86,86 %** |
| Taux Composite | 6,11 % |
| Taux Répondants | 16,80 % |
| PDTS moyen | 6,6 |
| Enquêtes | 839 |

**Axes de challenge** :
- Réduire les PDTS : transformer 2-3 détracteurs en satisfaits
- Capitaliser sur leur bon taux de répondants pour amplifier les 4/4
- Objectif score : **dépasser les 90%**

---

### Groupe 4 — ⚡ Performance Moyenne / Grand Volume (7 BOs)

> **Profil** : BOs à fort volume d'enquêtes avec un score intermédiaire (78-85%), un nombre élevé de PDTS (11-16). Le volume important crée un levier significatif sur le score global.

| Base opérationnelle | Score Satisfaction | Taux Composite | PDTS | Enquêtes | Taux Répondants | Poids Global |
|---|---|---|---|---|---|---|
| **BO Roannais** | 85,31 % | 4,63 % | 16 | 1 411 | 17,36 % | 1,73 % |
| **BO Forez** | 85,02 % | 3,93 % | 14 | 1 609 | 15,35 % | 1,74 % |
| **BO Montélimar** | 82,91 % | 6,05 % | 11 | 1 251 | 15,91 % | 1,21 % |
| **BO l'Arbresle** | 80,00 % | 8,90 % | 11 | 1 126 | 14,65 % | 0,87 % |
| **BO Heyrieux** | 78,61 % | 9,13 % | 16 | 1 106 | 16,91 % | 1,02 % |
| **BO Romans** | 78,43 % | 14,61 % | 12 | 990 | 15,45 % | 0,75 % |
| **BO Gleize** | 78,39 % | 7,16 % | 16 | 1 324 | 15,03 % | 1,12 % |

| Indicateur | Moyenne du groupe |
|---|---|
| Score Satisfaction | **81,24 %** |
| Taux Composite | 7,77 % |
| Taux Répondants | 15,81 % |
| PDTS moyen | 13,7 |
| Enquêtes | 1 260 |

**Axes de challenge** :
- **Priorité n°1** : Réduire les PDTS — chaque PDTS évité a un impact fort vu le volume
- Augmenter les 4/4 en travaillant le parcours client
- Objectif score : **dépasser les 85%**

---

### Groupe 5 — 🔧 Performance à Renforcer / Petit Volume (3 BOs)

> **Profil** : BOs avec un score faible (74-79%), un taux de répondants bas (~13%) et un volume d'enquêtes réduit. Ont besoin d'un effort combiné sur engagement et qualité.

| Base opérationnelle | Score Satisfaction | Taux Composite | PDTS | Enquêtes | Taux Répondants | Poids Global |
|---|---|---|---|---|---|---|
| **BO Rillieux** | 78,95 % | 4,44 % | 8 | 873 | 10,88 % | 0,37 % |
| **BO Vénissieux** | 76,04 % | 12,98 % | 10 | 785 | 12,23 % | 0,36 % |
| **BO Firminy - Saint Bonnet** | 74,42 % | 8,05 % | 10 | 566 | 15,19 % | 0,29 % |

| Indicateur | Moyenne du groupe |
|---|---|
| Score Satisfaction | **76,47 %** |
| Taux Composite | 8,49 % |
| Taux Répondants | 12,77 % |
| PDTS moyen | 9,3 |
| Enquêtes | 741 |

**Axes de challenge** :
- **Priorité n°1** : Augmenter massivement le taux de répondants (10,9% → 15%+) pour diluer le poids des PDTS
- Réduire le nombre de PDTS de 9 → 6
- Objectif score : **dépasser les 80%**

---

### Groupe 6 — 🔧 Performance à Renforcer / Grand Volume (4 BOs)

> **Profil** : BOs à fort volume avec le plus grand nombre de PDTS (14-17) et un score bas (72-80%). L'enjeu est majeur car leur poids dans le score global est significatif.

| Base opérationnelle | Score Satisfaction | Taux Composite | PDTS | Enquêtes | Taux Répondants | Poids Global |
|---|---|---|---|---|---|---|
| **BO Bourg - Montrevel** | 80,49 % | 10,91 % | 17 | 1 489 | 13,77 % | 1,21 % |
| **BO Saint Étienne** | 77,14 % | 11,47 % | 14 | 1 157 | 12,10 % | 0,64 % |
| **BO Givors** | 73,91 % | 12,85 % | 17 | 1 098 | 12,57 % | 0,59 % |
| **BO Oullins** | 72,14 % | 12,24 % | 17 | 1 064 | 13,16 % | 0,58 % |

| Indicateur | Moyenne du groupe |
|---|---|
| Score Satisfaction | **75,92 %** |
| Taux Composite | 11,87 % |
| Taux Répondants | 12,90 % |
| PDTS moyen | 16,2 |
| Enquêtes | 1 202 |

**Axes de challenge** :
- **Priorité n°1** : Réduire drastiquement les PDTS (16 → 10) — impact maximal sur le score
- **Priorité n°2** : Augmenter le taux de répondants (13% → 16%) pour booster les 4/4 et 3/4
- Analyser les causes racines des insatisfactions (volume élevé = échantillon riche pour l'analyse)
- Objectif score : **dépasser les 78%**

---

## Synthèse des Groupes

| Groupe | Nb BOs | Score Moyen | PDTS Moyen | Enquêtes Moy. | Taux Rép. | Objectif Score |
|---|---|---|---|---|---|---|
| 1 - Haute Perf. / Petit Vol. | 1 | 95,00 % | 0 | 285 | 14,0 % | Maintien |
| 2 - Perf. Int.-Haute / Vol. Moyen | 5 | 87,49 % | 6,8 | 1 099 | 13,4 % | > 90 % |
| 3 - Perf. Int.-Haute / Vol. Moyen | 5 | 86,86 % | 6,6 | 839 | 16,8 % | > 90 % |
| 4 - Perf. Moyenne / Grand Vol. | 7 | 81,24 % | 13,7 | 1 260 | 15,8 % | > 85 % |
| 5 - Perf. à Renforcer / Petit Vol. | 3 | 76,47 % | 9,3 | 741 | 12,8 % | > 80 % |
| 6 - Perf. à Renforcer / Grand Vol. | 4 | 75,92 % | 16,2 | 1 202 | 12,9 % | > 78 % |

## Leviers principaux du challenge

### 1. Augmenter le taux de répondants
Plus de répondants = plus de chance d'avoir des 4/4 et 3/4, ce qui **augmente mécaniquement** le nouveau score.

### 2. Réduire les PDTS (réponses 1/4)
Chaque PDTS en moins a un **double effet** :
- Baisse du taux de 1/4 (qui est soustrait dans le nouveau calcul)
- Augmentation relative des taux de 3/4 et 4/4

### 3. Convertir les neutres (2/4) en satisfaits (3/4 ou 4/4)
Les 2/4 sont « neutres » dans le nouveau calcul mais représentent un potentiel de conversion.

---

## Fichiers produits

| Fichier | Description |
|---|---|
| `analyse_groupes_bo.py` | Script Python d'analyse et de clustering |
| `groupes_bo_comparables.csv` | Export CSV des groupes avec toutes les métriques |
| `01_groupes_bo_scatter.png` | Carte des groupes (Score vs Volume) |
| `02_scores_par_groupe.png` | Scores par BO au sein de chaque groupe |
| `03_heatmap_groupes.png` | Profil moyen par groupe (heatmap) |
| `04_analyse_detaillee.png` | 4 vues d'analyse croisée |
