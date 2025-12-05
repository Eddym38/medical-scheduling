"""
Point d'entrée pour lancer la simulation multi-agent.
"""
from model import SchedulingModel
import sys
from pathlib import Path
from data.generator import load_competence_matrix

# Charger la matrice par défaut
competence_matrix = load_competence_matrix("competence_matrix_100.json")

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
        n_tabu=1,          # 1 agent Tabu
        collaboratif=True  # Mode collaboratif activé

    )

    print(f"Agents créés: {len(model.my_agents)}")
    for i, agent in enumerate(model.my_agents):
        agent_type = type(agent).__name__
        print(f"  - Agent {i+1}: {agent_type}")
    print(f"Étapes de simulation: {n_steps}")
    print()

    # Exécuter la simulation
    for step in range(n_steps):
        model.step()

        if step % 10 == 0:
            # Afficher les makespans de chaque agent
            agents_info = []
            for agent in model.my_agents:
                if hasattr(agent, 'makespan'):
                    agents_info.append(
                        f"{type(agent).__name__[:3]}:{agent.makespan}")
                elif hasattr(agent, 'best_makespan'):
                    agents_info.append(
                        f"{type(agent).__name__[:3]}:{agent.best_makespan}")

            agents_str = " | ".join(agents_info) if agents_info else "N/A"
            print(
                f"Étape {step:3d} | Global: {model.global_best_makespan} | Agents: {agents_str}")

    print()
    print("=" * 70)
    print(" " * 20 + "RÉSULTATS FINAUX")
    print("=" * 70)
    print(f"Meilleur CMax global: {model.global_best_makespan}")
    print()
    print("Performance par agent:")
    for i, agent in enumerate(model.my_agents):
        agent_type = type(agent).__name__
        if hasattr(agent, 'makespan'):
            print(f"  - {agent_type}: {agent.makespan}")
        elif hasattr(agent, 'best_makespan'):
            print(f"  - {agent_type}: {agent.best_makespan}")

    # Récupérer les données collectées
    data = model.datacollector.get_model_vars_dataframe()
    print(f"\nDonnées collectées: {len(data)} étapes")

    return model


if __name__ == "__main__":
    model = run_simulation(n_steps=100)
