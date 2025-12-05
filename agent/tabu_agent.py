"""
Agent utilisant la recherche Tabu.
"""
from mesa import Agent


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
