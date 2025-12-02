# Unification de l'Architecture

## Objectif

Tous les algorithmes retournent maintenant une **liste d'opérations ordonnées** et utilisent un **décodeur commun**.

## Architecture Unifiée

### Modules Partagés

1. **`src/decoding/decoder.py`**

   - `decode_chromosome(chromosome, competence_matrix)` : Convertit une liste d'opérations `[[patient, op], ...]` en matrice de planning

2. **`src/evaluation/fitness.py`**

   - `calculate_makespan(solution)` : Calcule le CMax à partir d'une solution décodée
   - `fitness(chromosome, competence_matrix, decode_fn)` : Wrapper générique pour évaluation

3. **`src/visualization/display.py`**
   - `plot_planning(solution, title, cmax)` : Affichage uniforme pour tous les algorithmes

### Algorithmes

#### 1. Algorithme Génétique (`genetic.py`)

**Format de sortie** : Liste d'opérations ordonnées `[[patient, op], ...]`

```python
from src.decoding.decoder import decode_chromosome
from src.evaluation.fitness import calculate_makespan

# Retourne directement le chromosome (liste d'opérations)
best_chrom = [[0,0], [1,0], [0,1], ...]
solution = decode_chromosome(best_chrom, competence_matrix)
cmax = calculate_makespan(solution)
```

**Statut** : ✅ Déjà compatible

---

#### 2. Recuit Simulé (`simulated_annealing.py`)

**Format de sortie** : Liste d'opérations ordonnées `[[patient, op], ...]`

**Modifications apportées** :

- ✅ Suppression de `reconstruire_solution()` (remplacé par `decode_chromosome`)
- ✅ Suppression de `calculer_cmax()` (remplacé par `calculate_makespan`)
- ✅ `algo_rs()` retourne seulement `ordre_star` (la liste d'opérations)
- ✅ `generate_solution_voisine()` retourne seulement `nouvel_ordre`

```python
from src.decoding.decoder import decode_chromosome
from src.evaluation.fitness import calculate_makespan

# Retourne la liste d'opérations
ordre_star = algo_rs(competence_matrix, ...)
solution = decode_chromosome(ordre_star, competence_matrix)
cmax = calculate_makespan(solution)
```

**Statut** : ✅ Converti

---

#### 3. Recherche Tabu (`tabu.py`)

**Format de sortie** : Liste d'opérations ordonnées `[[patient, op], ...]`

**Modifications apportées** :

- ✅ Ajout de `schedule_to_ordered_operations()` : Convertit le schedule interne en liste d'opérations
- ✅ Le `__main__` utilise maintenant `decode_chromosome` et `calculate_makespan`
- ⚠️ **Note** : L'algorithme utilise toujours son format interne `Dict[skill, List[Tuple]]` pendant l'optimisation, mais expose une interface uniforme

**Conversion** :

```python
from src.decoding.decoder import decode_chromosome
from src.evaluation.fitness import calculate_makespan

# L'algorithme retourne un schedule (format interne)
best_schedule, best_cmax, iterations, history = tabu_search(...)

# Conversion vers format unifié
ordered_operations = schedule_to_ordered_operations(best_schedule, C_array)
solution = decode_chromosome(ordered_operations, C_array)
cmax = calculate_makespan(solution)
```

**Statut** : ✅ Converti avec wrapper de conversion

---

## Tests de Validation

Tous les algorithmes ont été testés avec succès :

### Génétique

```
Makespan (CMax) : 40
Temps d'exécution : 0.82 secondes
```

### Recuit Simulé

```
Makespan (CMax) : 39
Temps d'exécution : 18.23 secondes
```

### Tabu

```
Makespan (CMax) : 43
Temps d'exécution : 4.23 secondes
```

## Format de Données

### Entrée

- **Matrice de compétences** : `competence_matrix[patient][operation] = duration`
- Chargée depuis `src/data/competence_matrix.json`

### Sortie Unifiée

Tous les algorithmes utilisent le même pipeline :

1. **Liste d'opérations** : `[[patient_id, operation_id], ...]`
   - Exemple : `[[0,0], [1,0], [0,1], [2,0], ...]`
2. **Solution décodée** : `solution[skill][time] = [patient, op] ou None`
   - Matrice 2D représentant le planning par compétence
3. **Makespan** : `int` - durée totale maximale

## Avantages

✅ **Réutilisabilité** : Fonctions de décodage et d'évaluation partagées  
✅ **Cohérence** : Format de sortie identique pour tous les algorithmes  
✅ **Maintenabilité** : Corrections de bugs centralisées  
✅ **Testabilité** : Interface uniforme facilite les tests comparatifs  
✅ **Extensibilité** : Nouveaux algorithmes peuvent facilement adopter cette architecture
