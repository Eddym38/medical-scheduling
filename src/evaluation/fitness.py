"""
Module d'évaluation des solutions.
"""


def calculate_makespan(solution):
    """
    Calcule le makespan (CMax) d'une solution décodée.

    Args:
        solution: Matrice [skill][time] avec (patient, op_idx) ou None

    Returns:
        int: Le makespan (temps total de la planification)
    """
    if not solution or not solution[0]:
        return 0
    return len(solution[0])


def fitness(chromosome, competence_matrix, decode_fn):
    """
    Calcule la fitness d'un chromosome.

    Args:
        chromosome: Liste de [patient, op_idx]
        competence_matrix: Matrice des compétences
        decode_fn: Fonction de décodage à utiliser

    Returns:
        float: Valeur de fitness (1 / (1 + makespan))
    """
    solution = decode_fn(chromosome, competence_matrix)
    makespan = calculate_makespan(solution)
    return 1 / (1 + makespan)
