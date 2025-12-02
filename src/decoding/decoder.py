"""
Module de décodage des chromosomes en solutions planifiées.
"""


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
