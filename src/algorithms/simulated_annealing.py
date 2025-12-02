import math
import random

from src.decoding.decoder import decode_chromosome
from src.evaluation.fitness import calculate_makespan


def extraire_liste_operations(tableau_patients):
    operations = []
    for patient in range(len(tableau_patients)):
        for ope in range(len(tableau_patients[patient])):
            if any(comp > 0 for comp in tableau_patients[patient][ope]):
                operations.append([patient, ope])
    return operations


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


def algo_rs(competence_matrix, t0=100, alpha=0.95, nbiter_cycle=30):
    ordre = extraire_liste_operations(competence_matrix)
    ordre_star = ordre.copy()
    
    solution = decode_chromosome(ordre, competence_matrix)
    cmax_star = calculate_makespan(solution)

    t = t0
    nouveau_cycle = True

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
    from src.utils.common import run_and_display

    def wrapper_simulated_annealing(comp_matrix, **params):
        ordre = algo_rs(comp_matrix, **params)
        solution = decode_chromosome(ordre, comp_matrix)
        cmax = calculate_makespan(solution)
        return ordre, solution, cmax

    run_and_display(
        algorithm_name="Recuit Simulé",
        algorithm_func=wrapper_simulated_annealing,
        competence_matrix=competence_matrix,
        plot_func=plot_planning,
        calculate_makespan_func=calculate_makespan,
        t0=100,
        alpha=0.95,
        nbiter_cycle=30
    )
