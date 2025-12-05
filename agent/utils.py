"""
Utilitaires pour les agents.
Fonctions communes utilisées par tous les agents.
"""
from src.utils.population import create_random_solution
from src.evaluation.fitness import calculate_makespan
from src.decoding.decoder import decode_chromosome
import sys
from pathlib import Path

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))


def evaluate_solution(operations, competence_matrix):
    """
    Évalue une solution (liste d'opérations) et retourne son makespan.

    Args:
        operations: Liste d'opérations [[patient, op], ...]
        competence_matrix: Matrice de compétences

    Returns:
        int: Makespan de la solution
    """
    solution = decode_chromosome(operations, competence_matrix)
    makespan = calculate_makespan(solution)
    return makespan


def generate_initial_solution(competence_matrix):
    """
    Génère une solution initiale aléatoire.

    Args:
        competence_matrix: Matrice de compétences

    Returns:
        Liste d'opérations [[patient, op], ...]
    """
    return create_random_solution(competence_matrix)
