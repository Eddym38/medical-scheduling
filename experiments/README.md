# Expérimentations - Tests des Algorithmes

Ce dossier contient les scripts de test pour évaluer les trois algorithmes d'ordonnancement avec différentes configurations et tailles de problèmes.

## Structure

```
experiments/
├── run_all_tests.py           # Script principal pour lancer tous les tests
├── test_genetic.py            # Tests de l'algorithme génétique
├── test_simulated_annealing.py # Tests du recuit simulé
├── test_tabu.py               # Tests de la recherche Tabu
├── analyze_results.py         # Analyse et comparaison des résultats
└── results/                   # Dossier contenant les résultats (généré)
    ├── genetic_results.json
    ├── simulated_annealing_results.json
    └── tabu_results.json
```

## Utilisation

### 1. Générer les matrices de compétences de test

Avant de lancer les tests, générez les matrices de différentes tailles :

```bash
python -m src.data.generator
```

Cela créera 4 fichiers dans `src/data/` :

- `competence_matrix_10.json` (10 patients)
- `competence_matrix_50.json` (50 patients)
- `competence_matrix_100.json` (100 patients)
- `competence_matrix.json` (20 patients - par défaut)

### 2. Lancer tous les tests

Pour exécuter tous les tests d'un coup :

```bash
cd experiments
python run_all_tests.py
```

Cette commande :

- Vérifie la présence des matrices de test
- Lance les tests pour les 3 algorithmes
- Sauvegarde les résultats dans `results/`

### 3. Lancer les tests individuellement

Vous pouvez aussi tester chaque algorithme séparément :

```bash
cd experiments

# Test algorithme génétique
python test_genetic.py

# Test recuit simulé
python test_simulated_annealing.py

# Test recherche Tabu
python test_tabu.py
```

### 4. Analyser les résultats

Pour comparer les performances des algorithmes :

```bash
cd experiments
python analyze_results.py
```

Cela affichera :

- Résultats détaillés pour chaque algorithme
- Meilleurs paramètres pour chaque taille de problème
- Comparaison globale des trois algorithmes

## Configuration des tests

### Algorithme Génétique (`test_genetic.py`)

Teste différentes **tailles de population** :

- **10 patients** : pop ∈ {20, 50, 100, 150}
- **50 patients** : pop ∈ {50, 100, 200, 300}
- **100 patients** : pop ∈ {100, 200, 300, 500}

Paramètres fixes : 100 générations

### Recuit Simulé (`test_simulated_annealing.py`)

Teste différentes **températures initiales** :

- **10 patients** : T ∈ {50, 100, 200, 500}, 500 itérations
- **50 patients** : T ∈ {100, 200, 500, 1000}, 1000 itérations
- **100 patients** : T ∈ {200, 500, 1000, 2000}, 2000 itérations

Paramètre fixe : α = 0.95

### Recherche Tabu (`test_tabu.py`)

Teste différentes **tenures** (durée tabu) :

- **10 patients** : tenure ∈ {3, 5, 7, 10}, 300 itérations, 30 candidats
- **50 patients** : tenure ∈ {5, 7, 10, 15}, 500 itérations, 50 candidats
- **100 patients** : tenure ∈ {7, 10, 15, 20}, 600 itérations, 70 candidats

## Format des résultats

Les résultats sont sauvegardés au format JSON avec les informations suivantes :

### Génétique

```json
{
  "matrix_file": "competence_matrix_50.json",
  "num_patients": 50,
  "total_operations": 250,
  "pop_size": 100,
  "num_generations": 100,
  "cmax": 42,
  "fitness": 0.02381,
  "time": 12.34
}
```

### Recuit Simulé

```json
{
  "matrix_file": "competence_matrix_50.json",
  "num_patients": 50,
  "total_operations": 250,
  "temperature": 500.0,
  "max_iterations": 1000,
  "cmax": 41,
  "time": 18.56
}
```

### Tabu

```json
{
  "matrix_file": "competence_matrix_50.json",
  "num_patients": 50,
  "total_operations": 250,
  "tenure": 7,
  "max_iterations": 500,
  "candidate_size": 50,
  "cmax": 43,
  "iterations_done": 482,
  "time": 8.92
}
```

## Personnalisation

Pour modifier les paramètres testés, éditez directement les fichiers de test :

- `test_genetic.py` : modifiez `pop_sizes_configs` et `num_generations`
- `test_simulated_annealing.py` : modifiez `temp_configs` et `iterations_configs`
- `test_tabu.py` : modifiez `tenure_configs` et `iteration_configs`

## Exemples de sortie

### Résumé des tests

```
==============================================================================
                         TESTS ALGORITHME GENETIQUE
==============================================================================

======================================================================
Test sur: competence_matrix_10.json
======================================================================
Patients: 10 | Operations totales: 30
Generations: 100

Population:   20 | CMax:  18 | Fitness: 0.055556 | Temps:   1.23s
Population:   50 | CMax:  17 | Fitness: 0.058824 | Temps:   2.45s
Population:  100 | CMax:  16 | Fitness: 0.062500 | Temps:   4.12s
Population:  150 | CMax:  16 | Fitness: 0.062500 | Temps:   5.89s
```

### Analyse comparative

```
==============================================================================
                        COMPARAISON DES ALGORITHMES
==============================================================================

Matrice                        | Algorithme           |  CMax | Temps (s)
--------------------------------------------------------------------------------
competence_matrix_10.json      | Genetique            |    16 |       4.12
                               | Recuit Simule        |    15 |       3.56
                               | Tabu                 |    17 |       2.34
                               | ** Meilleur: Recuit Simule (CMax=15) **
```

## Notes

- Les tests peuvent prendre du temps, surtout pour les grandes instances
- Les résultats peuvent varier selon le seed aléatoire
- Pour des résultats reproductibles, fixez le seed dans chaque algorithme
