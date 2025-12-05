"""
Agent utilisant le recuit simulé.
"""
from mesa import Agent


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
