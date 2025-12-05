import math
import random

from src.decoding.decoder import decode_chromosome
from src.evaluation.fitness import calculate_makespan
from src.utils.population import create_random_solution


def verifier_precedence(ordre_operations):
    dernieres_ops = {}
    for patient_id, op_id in ordre_operations:
        if patient_id in dernieres_ops and op_id <= dernieres_ops[patient_id]:
            return False
        dernieres_ops[patient_id] = op_id
    return True


def generate_solution_voisine(tableau_patients, ordre_actuel):
    max_tentatives = 100
    for _ in range(max_tentatives):
        nouvel_ordre = ordre_actuel.copy()
        i, j = random.sample(range(len(nouvel_ordre)), 2)
        if nouvel_ordre[i][0] != nouvel_ordre[j][0]:
            nouvel_ordre[i], nouvel_ordre[j] = nouvel_ordre[j], nouvel_ordre[i]
            if verifier_precedence(nouvel_ordre):
                return nouvel_ordre
    return ordre_actuel


def algo_rs(competence_matrix, t0=100, alpha=0.95, nbiter_cycle=30, initial_solution=None, verbose=True):
    """
    Algorithme de recuit simulé pour l'ordonnancement.

    Args:
        competence_matrix: Matrice de compétences
        t0: Température initiale
        alpha: Coefficient de refroidissement
        nbiter_cycle: Nombre d'itérations par cycle
        initial_solution: Solution initiale optionnelle [[patient, op], ...]
        verbose: Affichage des informations

    Returns:
        Liste d'opérations ordonnées [[patient, op], ...]
    """
    if initial_solution is not None:
        ordre = [op.copy() for op in initial_solution]
    else:
        ordre = create_random_solution(competence_matrix)

    ordre_star = ordre.copy()

    solution = decode_chromosome(ordre, competence_matrix)
    cmax_star = calculate_makespan(solution)

    t = t0
    nouveau_cycle = True

    if verbose:
        print(f"Solution initiale Cmax : {cmax_star}")
        print(f"Nombre d'opérations : {len(ordre)}\n")

    while nouveau_cycle:
        nbiter = 0
        nouveau_cycle = False

        while nbiter < nbiter_cycle:
            nbiter += 1
            ordre_prime = generate_solution_voisine(competence_matrix, ordre)

            solution_prime = decode_chromosome(ordre_prime, competence_matrix)
            cmax_prime = calculate_makespan(solution_prime)

            solution = decode_chromosome(ordre, competence_matrix)
            cmax = calculate_makespan(solution)

            delta_f = cmax_prime - cmax

            if delta_f < 0 or (t > 1e-8 and random.random() < math.exp(-delta_f / t)):
                ordre = ordre_prime.copy()
                nouveau_cycle = True

            solution_current = decode_chromosome(ordre, competence_matrix)
            cmax_current = calculate_makespan(solution_current)
            if cmax_current < cmax_star:
                ordre_star = ordre.copy()
                cmax_star = cmax_current

        t *= alpha

    return ordre_star


if __name__ == "__main__":
    from src.data.instances import competence_matrix
    from src.visualization.display import plot_planning

    print("="*60)
    print(" "*15 + "RECUIT SIMULE")
    print("="*60)
    print()

    import time
    start_time = time.time()

    best_operations = algo_rs(
        competence_matrix=competence_matrix,
        t0=100,
        alpha=0.95,
        nbiter_cycle=30,
        verbose=True
    )

    elapsed = time.time() - start_time

    # Décoder et afficher
    best_solution = decode_chromosome(best_operations, competence_matrix)
    best_makespan = calculate_makespan(best_solution)

    print()
    print("="*60)
    print(" "*15 + "RESULTATS FINAUX")
    print("="*60)
    print(f"Makespan (CMax) : {best_makespan}")
    print(f"Temps d'execution : {elapsed:.2f} secondes")

    plot_planning(
        best_solution, title=f"Recuit Simule (CMax = {best_makespan})")
