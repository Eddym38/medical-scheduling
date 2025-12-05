"""
Agents du système multi-agent.
Chaque agent représente un algorithme d'optimisation.
"""
from mesa import Agent


class GeneticAgent(Agent):
    """Agent utilisant l'algorithme génétique."""

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.current_solution = None
        self.best_makespan = float('inf')

    def step(self):
        """Une étape d'exécution de l'agent."""
        # TODO: Implémenter la logique de l'agent génétique
        pass


class SimulatedAnnealingAgent(Agent):
    """Agent utilisant le recuit simulé."""

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.current_solution = None
        self.best_makespan = float('inf')
        self.temperature = 100.0

    def step(self):
        """Une étape d'exécution de l'agent."""
        # TODO: Implémenter la logique du recuit simulé
        pass


class TabuAgent(Agent):
    """Agent utilisant la recherche Tabu."""

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.current_solution = None
        self.best_makespan = float('inf')
        self.tabu_list = []

    def step(self):
        """Une étape d'exécution de l'agent."""
        # TODO: Implémenter la logique de la recherche Tabu
        pass
