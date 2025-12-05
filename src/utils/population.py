"""
Utilitaires pour la génération de solutions initiales.
Partagé par tous les algorithmes pour assurer la cohérence.
"""
import random


def create_random_solution(competence_matrix):
    """
    Crée une solution aléatoire (liste d'opérations ordonnées).

    Args:
        competence_matrix: Matrice de compétences [patient][operation] = duration

    Returns:
        Liste d'opérations [[patient, op_idx], ...] dans un ordre aléatoire
    """
    operations = []
    nb_patients = len(competence_matrix)
    for patient in range(nb_patients):
        for op_idx in range(len(competence_matrix[patient])):
            operations.append([patient, op_idx])
    random.shuffle(operations)
    return operations


def create_initial_population(population_size, competence_matrix):
    """
    Crée une population initiale de solutions aléatoires.

    Args:
        population_size: Nombre de solutions à générer
        competence_matrix: Matrice de compétences

    Returns:
        Liste de solutions (chaque solution est une liste d'opérations)
    """
    return [create_random_solution(competence_matrix) for _ in range(population_size)]


def extract_operations_list(competence_matrix):
    """
    Extrait la liste de toutes les opérations valides d'une matrice de compétences.

    Args:
        competence_matrix: Matrice de compétences

    Returns:
        Liste d'opérations [[patient, op_idx], ...] (non mélangée)
    """
    operations = []
    nb_patients = len(competence_matrix)
    for patient in range(nb_patients):
        for op_idx in range(len(competence_matrix[patient])):
            # Vérifier qu'au moins une compétence peut faire cette opération
            if any(comp > 0 for comp in competence_matrix[patient][op_idx]):
                operations.append([patient, op_idx])
    return operations
