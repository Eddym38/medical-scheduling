# Architecture du projet Medical Scheduling

## 📁 Structure modulaire

```
src/
├── algorithms/              # Algorithmes d'ordonnancement UNIQUEMENT
│   ├── genetic.py          # Algorithme génétique
│   ├── simulated_annealing.py  # Recuit simulé
│   └── tabu.py             # Recherche tabou
│
├── data/                    # Gestion des données
│   ├── generator.py        # Génération et sauvegarde de matrices
│   └── instances.py        # Chargement de la matrice par défaut
│
├── decoding/               # Décodage de solutions
│   └── decoder.py          # decode_chromosome()
│
├── evaluation/             # Évaluation de solutions
│   └── fitness.py          # calculate_makespan(), fitness()
│
├── visualization/          # Visualisation
│   └── display.py          # plot_planning()
│
└── utils/                  # Utilitaires communs
    └── common.py           # run_and_display()
```

## 🎯 Fonctions communes réutilisables

### `src/evaluation/fitness.py`

- `calculate_makespan(solution)` → Calcule le CMax depuis une solution décodée
- `fitness(chromosome, competence_matrix, decode_fn)` → Calcule la fitness

### `src/decoding/decoder.py`

- `decode_chromosome(chromosome, competence_matrix)` → Décode en matrice de solution

### `src/visualization/display.py`

- `plot_planning(solution, title, save_path)` → Affiche le planning (sans calcul)

### `src/utils/common.py`

- `run_and_display(...)` → Exécute un algo et affiche les résultats

## 🚀 Utilisation

### Générer et sauvegarder une matrice (UNE FOIS)

```bash
python -m src.data.generator
```

### Exécuter un algorithme

```bash
python -m src.algorithms.genetic
python -m src.algorithms.simulated_annealing
python -m src.algorithms.tabu
```

### Dans votre code

```python
from src.decoding.decoder import decode_chromosome
from src.evaluation.fitness import calculate_makespan
from src.visualization.display import plot_planning

# Décoder
solution = decode_chromosome(chromosome, competence_matrix)

# Calculer le CMax SANS plotter
cmax = calculate_makespan(solution)

# Afficher si nécessaire
plot_planning(solution, title=f"Planning (CMax={cmax})")
```

## ✅ Avantages

1. **Séparation des responsabilités** : Chaque module a un rôle clair
2. **Réutilisabilité** : Toutes les fonctions communes sont partagées
3. **Testabilité** : Chaque fonction peut être testée indépendamment
4. **Simplicité** : Les algorithmes ne contiennent que la logique algorithmique
