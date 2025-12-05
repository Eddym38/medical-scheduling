"""
Imports des agents pour faciliter l'utilisation.
Chaque agent est maintenant dans son propre fichier.
"""
from .genetic_agent import GeneticAgent
from .simulated_annealing_agent import SimulatedAnnealingAgent
from .tabu_agent import TabuAgent

__all__ = ['GeneticAgent', 'SimulatedAnnealingAgent', 'TabuAgent']
