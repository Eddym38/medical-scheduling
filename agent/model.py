"""
Modèle Mesa pour la simulation multi-agent.
Coordonne les agents et gère l'environnement partagé.
"""
from mesa import Model
from mesa.datacollection import DataCollector
# Créer les agents
from genetic_agent import GeneticAgent
from simulated_annealing_agent import SimulatedAnnealingAgent
from tabu_agent import TabuAgent


class SchedulingModel(Model):
    """Modèle de simulation pour l'ordonnancement multi-agent."""

    def __init__(self, competence_matrix, n_genetic=1, n_simulated=1, n_tabu=1, collaboratif=False):
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

        # Liste manuelle des agents (pas de scheduler)
        self.my_agents = []

        # Meilleure solution globale trouvée par tous les agents
        self.global_best_solution = None
        self.global_best_makespan = float('inf')

        # Agents génétiques
        for i in range(n_genetic):
            agent = GeneticAgent(
                model=self, inner_population_size=20, collaboratif=collaboratif, mutation_rate=0.2)
            self.my_agents.append(agent)

        # Agents recuit simulé
        for i in range(n_simulated):
            agent = SimulatedAnnealingAgent(self, collaboratif=collaboratif)
            self.my_agents.append(agent)

        # Agents Tabu
        for i in range(n_tabu):
            agent = TabuAgent(self, collaboratif=collaboratif)
            self.my_agents.append(agent)

        # Collecteur de données
        self.datacollector = DataCollector(
            model_reporters={
                "Best_Makespan": lambda m: m.global_best_makespan,
            }
        )

    def step(self):
        """Une étape de simulation - exécute step() de tous les agents."""
        # Exécuter manuellement chaque agent
        for agent in self.my_agents:
            agent.step()
            self.update_global_best(agent.best_order, agent.makespan)

        # Collecter les données
        self.datacollector.collect(self)

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
