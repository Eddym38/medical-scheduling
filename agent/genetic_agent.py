"""
Agent utilisant l'algorithme génétique.
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
