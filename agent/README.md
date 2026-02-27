# Système Multi-Agent avec Mesa

Structure du dossier `agent/` pour l'ordonnancement médical avec Mesa.

## Architecture

```
agent/
├── __init__.py                      # Package initialization
├── agents.py                        # Imports centralisés des agents
├── genetic_agent.py                 # Agent algorithme génétique
├── simulated_annealing_agent.py     # Agent recuit simulé
├── tabu_agent.py                    # Agent recherche Tabu
├── model.py                         # Modèle Mesa (SchedulingModel)
├── utils.py                         # Fonctions utilitaires partagées
├── main.py                          # Point d'entrée pour lancer la simulation
└── README.md                        # Cette documentation
```

## Fichiers

### `genetic_agent.py`

Agent utilisant l'algorithme génétique.

- Classe: `GeneticAgent`
- **Binôme responsable:** [À remplir]

### `simulated_annealing_agent.py`

Agent utilisant le recuit simulé.

- Classe: `SimulatedAnnealingAgent`
- **Binôme responsable:** [À remplir]

### `tabu_agent.py`

Agent utilisant la recherche Tabu.

- Classe: `TabuAgent`
- **Binôme responsable:** [À remplir]

### `agents.py`

Fichier d'imports centralisés pour faciliter l'utilisation des agents.

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

Pour travailler sur votre agent :

1. **Ouvrez le fichier correspondant à votre agent** :

   - `genetic_agent.py` pour l'algorithme génétique
   - `simulated_annealing_agent.py` pour le recuit simulé
   - `tabu_agent.py` pour la recherche Tabu

2. **Implémentez la méthode `step()`** avec votre logique

3. **Utilisez les utilitaires** dans `utils.py` pour évaluer vos solutions

4. **Testez votre agent** en modifiant `main.py`

Chaque binôme travaille sur son propre fichier d'agent sans conflit !

## SMA Metier (Accueil / Identificateur / Ordonnanceur)

Le notebook `systeme_multi_agents.ipynb` est maintenant porte en Python executable :

- `sma.py` : structures de donnees, agents metier, sous-agents SA/Tabou/GA, coordinateur
- `sma_main.py` : point d'entree CLI pour lancer un scenario complet

Execution rapide :

```bash
python agent/sma_main.py --matrix competence_matrix_10.json --mode pipeline --steps-par-phase 50 --show-summary
```

Scenario avec absence de personnel :

```bash
python agent/sma_main.py --matrix competence_matrix_10.json --absence-id 2 --show-summary
```
