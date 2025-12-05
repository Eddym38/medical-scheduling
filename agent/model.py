"""
Modèle Mesa pour la simulation multi-agent.
Coordonne les agents et gère l'environnement partagé.
"""
from mesa import Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector


class SchedulingModel(Model):
    """Modèle de simulation pour l'ordonnancement multi-agent."""

    def __init__(self, competence_matrix, n_genetic=1, n_simulated=1, n_tabu=1):
        """
        Initialise le modèle.

        Args:
            competence_matrix: Matrice de compétences du problème
            n_genetic: Nombre d'agents génétiques
            n_simulated: Nombre d'agents recuit simulé
            n_tabu: Nombre d'agents Tabu
        """
        super().__init__()
        self.competence_matrix = competence_matrix
        self.schedule = RandomActivation(self)

        # Meilleure solution globale trouvée par tous les agents
        self.global_best_solution = None
        self.global_best_makespan = float('inf')

        # Créer les agents
        from .agents import GeneticAgent, SimulatedAnnealingAgent, TabuAgent

        agent_id = 0

        # Agents génétiques
        for i in range(n_genetic):
            agent = GeneticAgent(agent_id, self)
            self.schedule.add(agent)
            agent_id += 1

        # Agents recuit simulé
        for i in range(n_simulated):
            agent = SimulatedAnnealingAgent(agent_id, self)
            self.schedule.add(agent)
            agent_id += 1

        # Agents Tabu
        for i in range(n_tabu):
            agent = TabuAgent(agent_id, self)
            self.schedule.add(agent)
            agent_id += 1

        # Collecteur de données
        self.datacollector = DataCollector(
            model_reporters={
                "Best_Makespan": lambda m: m.global_best_makespan,
            }
        )

    def step(self):
        """Une étape de simulation."""
        self.datacollector.collect(self)
        self.schedule.step()

    def update_global_best(self, solution, makespan):
        """
        Met à jour la meilleure solution globale si nécessaire.

        Args:
            solution: Liste d'opérations
            makespan: Makespan de la solution

        Returns:
            True si c'est une amélioration
        """
        if makespan < self.global_best_makespan:
            self.global_best_solution = solution
            self.global_best_makespan = makespan
            return True
        return False
