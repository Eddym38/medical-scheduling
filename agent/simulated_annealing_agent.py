import math
import random

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



def decode_chromosome(chromosome, competence_matrix):
    """
    Décodage :
    - on lit les gènes dans l'ordre,
    - on ne planifie (p, op_idx) que si op_idx est la prochaine op pour ce patient,
    - dès qu'on planifie une opération, on repart du début du chromosome,
    - chaque bloc de skill_time est posé de manière consécutive sur le skill.

    Args:
        chromosome: Liste de [patient, op_idx] représentant l'ordre des opérations
        competence_matrix: Matrice des compétences [patient][operation][skill]

    Returns:
        solution: Matrice [skill][time] avec (patient, op_idx) ou None
    """
    nb_patient = len(competence_matrix)
    nb_skills = len(competence_matrix[0][0])

    solution = [[] for _ in range(nb_skills)]
    next_op = [0 for _ in range(nb_patient)]
    patient_end = [0 for _ in range(nb_patient)]

    total_ops = sum(len(competence_matrix[p]) for p in range(nb_patient))
    scheduled_ops = 0

    while scheduled_ops < total_ops:
        progress = False

        for patient, op_idx in chromosome:

            if op_idx < next_op[patient]:
                continue
            if op_idx > next_op[patient]:
                continue

            operation = competence_matrix[patient][op_idx]
            last_time_for_operation = patient_end[patient]

            for skill in range(nb_skills):
                skill_time = operation[skill]
                if skill_time <= 0:
                    continue

                t = patient_end[patient]

                while True:
                    current_len = len(solution[0]) if solution[0] else 0
                    while t + skill_time > current_len:
                        if not solution[0]:
                            for skill_row in solution:
                                skill_row.append(None)
                        else:
                            for skill_row in solution:
                                skill_row.append(None)
                        current_len += 1

                    bloc_libre = True
                    for tau in range(t, t + skill_time):
                        if solution[skill][tau] is not None:
                            bloc_libre = False
                            break

                    if bloc_libre:
                        for tau in range(t, t + skill_time):
                            solution[skill][tau] = (patient, op_idx)
                        fin_bloc = t + skill_time - 1
                        if fin_bloc > last_time_for_operation:
                            last_time_for_operation = fin_bloc
                        break
                    else:
                        t += 1

            patient_end[patient] = last_time_for_operation + 1
            next_op[patient] += 1
            scheduled_ops += 1
            progress = True
            break

        if not progress:
            # En cas de problème, retourne une solution vide
            return solution

    return solution


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



def algo_rs_step(best_order, makespan, competence_matrix, temperature, cooling_rate):
    """
    Une seule itération de l'algorithme de recuit simulé
    
    Args:
        best_order: ordre actuel des opérations [[patient_id, op_id], ...]
        makespan: makespan actuel
        competence_matrix: matrice de compétences
        temperature: température actuelle
        cooling_rate: taux de refroidissement
    
    Returns:
        tuple: (nouvel_ordre, nouveau_makespan, nouvelle_temperature)
    """
    # Génération d'une solution voisine
    ordre_prime = generate_solution_voisine(competence_matrix, best_order)
    
    # Calcul du makespan de la nouvelle solution
    solution_prime = decode_chromosome(ordre_prime, competence_matrix)
    cmax_prime = calculate_makespan(solution_prime)
    
    # Delta entre nouvelle et ancienne solution
    delta_f = cmax_prime - makespan
    
    # Critère d'acceptation de Metropolis
    if delta_f < 0 or (temperature > 1e-8 and random.random() < math.exp(-delta_f / temperature)):
        # Solution acceptée
        best_order = ordre_prime
        makespan = cmax_prime
    
    # Refroidissement
    temperature *= cooling_rate
    
    return best_order, makespan, temperature



import math
from mesa import Agent


class RecuitSimuleAgent(Agent):
    
    def simulated_annealing(self):
        """Algorithme de recuit simulé - une itération"""
        
        # Appel à l'algorithme de recuit simulé (une seule itération)
        self.best_order, self.makespan, self.temperature = algo_rs_step(
            self.best_order,
            self.makespan,
            self.model.competence_matrix,
            self.temperature,
            self.cooling_rate
        )
    
    def __init__(self, model, collaboratif=False, temp_init=1000, cooling_rate=0.95):
        super().__init__(model)
        
        # Initialiser l'ordre des opérations
        self.best_order = extraire_liste_operations(self.model.competence_matrix)
        
        # Calculer le makespan initial
        solution = decode_chromosome(self.best_order, self.model.competence_matrix)
        self.makespan = calculate_makespan(solution)
        
        # Paramètres du recuit simulé
        self.temperature = temp_init
        self.temp_init = temp_init
        self.cooling_rate = cooling_rate
        
        self.collaboratif = collaboratif
    
    def contact(self):
        '''
        si je suis collaboratif, j'entre en contact avec les autres
        je vérifie s'il y a mieux que moi, dans ce cas, je recupere le meilleur dans ma population
        '''
        min_makespan = self.makespan
        
        for a in self.model.agents:
            if a.makespan < min_makespan:
                self.makespan = a.makespan
                self.best_order = a.best_order.copy()
    
    def step(self):
        #print(f"I am n°{self.unique_id} and my makespan is : {self.makespan}")
        self.simulated_annealing()
        if self.collaboratif == True:
            self.contact()