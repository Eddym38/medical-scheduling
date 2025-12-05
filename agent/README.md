# Système Multi-Agent avec Mesa

Structure du dossier `agent/` pour l'ordonnancement médical avec Mesa.

## Architecture

```
agent/
├── __init__.py          # Package initialization
├── agents.py            # Définition des agents (GeneticAgent, SimulatedAnnealingAgent, TabuAgent)
├── model.py             # Modèle Mesa (SchedulingModel)
├── utils.py             # Fonctions utilitaires partagées
├── main.py              # Point d'entrée pour lancer la simulation
└── README.md            # Cette documentation
```

## Fichiers

### `agents.py`

Contient les classes d'agents :

- `GeneticAgent` - Agent utilisant l'algorithme génétique
- `SimulatedAnnealingAgent` - Agent utilisant le recuit simulé
- `TabuAgent` - Agent utilisant la recherche Tabu

Chaque agent hérite de `mesa.Agent` et implémente la méthode `step()`.

### `model.py`

Définit le modèle de simulation :

- `SchedulingModel` - Coordonne les agents
- Gère la meilleure solution globale
- Collecte les données de simulation

### `utils.py`

Fonctions utilitaires :

- `evaluate_solution()` - Évalue une solution
- `generate_initial_solution()` - Génère une solution aléatoire

### `main.py`

Point d'entrée pour lancer une simulation.

## Utilisation

```bash
# Installer Mesa
pip install mesa

# Lancer la simulation
python agent/main.py
```

## Exemple de code

```python
from agent.model import SchedulingModel
from src.data.instances import competence_matrix

# Créer le modèle
model = SchedulingModel(
    competence_matrix=competence_matrix,
    n_genetic=2,
    n_simulated=1,
    n_tabu=1
)

# Exécuter 100 étapes
for i in range(100):
    model.step()

print(f"Meilleur makespan: {model.global_best_makespan}")
```

## Extension

Pour ajouter un nouvel agent :

1. Créer une classe dans `agents.py` qui hérite de `Agent`
2. Implémenter la méthode `step()`
3. L'ajouter dans `model.py` lors de l'initialisation
