"""
Point d'entrée pour lancer la simulation multi-agent.
"""
from data import competence_matrix
from model import SchedulingModel
import sys
from pathlib import Path

# Ajouter le répertoire parent au path pour importer depuis src/
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_simulation(n_steps=100):
    """
    Lance une simulation multi-agent.

    Args:
        n_steps: Nombre d'étapes de simulation
    """
    print("=" * 70)
    print(" " * 20 + "SIMULATION MULTI-AGENT")
    print("=" * 70)
    print()

    # Créer le modèle avec les agents
    model = SchedulingModel(
        competence_matrix=competence_matrix,
        n_genetic=1,      # 1 agent génétique
        n_simulated=1,    # 1 agent recuit simulé
        n_tabu=1          # 1 agent Tabu
    )

    print(f"Agents créés: {len(model.my_agents)}")
    print(f"Étapes de simulation: {n_steps}")
    print()

    # Exécuter la simulation
    for step in range(n_steps):
        model.step()

        if step % 10 == 0:
            print(
                f"Étape {step:3d} | Meilleur CMax: {model.global_best_makespan}")

    print()
    print("=" * 70)
    print(" " * 20 + "RÉSULTATS FINAUX")
    print("=" * 70)
    print(f"Meilleur CMax trouvé: {model.global_best_makespan}")

    # Récupérer les données collectées
    data = model.datacollector.get_model_vars_dataframe()
    print(f"\nNombre de données collectées: {len(data)}")

    return model


if __name__ == "__main__":
    model = run_simulation(n_steps=100)
