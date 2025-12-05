

import random
from mesa import Agent


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


def create_random_chromosome(competence_matrix):
    """
    Crée un chromosome aléatoire (liste d'opérations ordonnées).

    Args:
        competence_matrix: Matrice de compétences [patient][operation][skill]

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


def apply_swap(chromosome, i, j):
    """
    Applique un swap entre deux positions dans le chromosome.

    Args:
        chromosome: Chromosome à modifier
        i, j: Indices à échanger

    Returns:
        Nouveau chromosome avec les positions i et j échangées
    """
    new_chromosome = [gene[:] for gene in chromosome]
    new_chromosome[i], new_chromosome[j] = new_chromosome[j], new_chromosome[i]
    return new_chromosome


def apply_insertion(chromosome, i, j):
    """
    Insère l'élément à la position i vers la position j.

    Args:
        chromosome: Chromosome à modifier
        i: Position de l'élément à déplacer
        j: Position de destination

    Returns:
        Nouveau chromosome avec l'élément déplacé
    """
    new_chromosome = [gene[:] for gene in chromosome]
    if i == j or len(new_chromosome) < 2:
        return new_chromosome

    elem = new_chromosome.pop(i)
    new_chromosome.insert(j, elem)
    return new_chromosome


def tabu_search_step(current_order, best_order, best_makespan, tabu_list,
                     iteration, competence_matrix, tenure=7, candidate_size=40):
    """
    Une seule itération de la recherche tabu sur chromosomes.

    Args:
        current_order: chromosome courant (liste de [patient, op_idx])
        best_order: meilleur chromosome trouvé
        best_makespan: makespan du meilleur chromosome
        tabu_list: liste des mouvements tabu [(i, j, iteration_expiration), ...]
        iteration: numéro d'itération actuel
        competence_matrix: matrice de compétences
        tenure: durée tabu
        candidate_size: nombre de candidats à générer

    Returns:
        tuple: (nouveau_current, nouveau_best, nouveau_best_makespan, nouveau_tabu_list)
    """
    candidates = []

    # Générer des voisins par insertion
    for _ in range(candidate_size):
        if len(current_order) < 2:
            continue

        i = random.randrange(len(current_order))
        j = random.randrange(len(current_order))

        if i == j:
            continue

        # Créer un voisin par insertion
        neighbor = apply_insertion(current_order, i, j)

        # Évaluer le voisin
        solution = decode_chromosome(neighbor, competence_matrix)
        makespan = calculate_makespan(solution)

        candidates.append((makespan, neighbor, (i, j)))

    if not candidates:
        return current_order, best_order, best_makespan, tabu_list

    # Trier les candidats par makespan
    candidates.sort(key=lambda x: x[0])

    # Nettoyer la liste tabu (retirer les mouvements expirés)
    tabu_list = [(i, j, exp) for i, j, exp in tabu_list if exp > iteration]

    # Choisir le meilleur candidat non-tabu (ou avec aspiration)
    chosen = None
    for makespan, neighbor, (i, j) in candidates:
        # Vérifier si le mouvement est tabu
        is_tabu = any((i == ti and j == tj) for ti, tj, _ in tabu_list)

        # Critère d'aspiration : accepter si meilleur que le best global
        if (not is_tabu) or (makespan < best_makespan):
            chosen = (makespan, neighbor, (i, j))
            break

    if chosen is None:
        chosen = candidates[0]

    makespan, neighbor, (i, j) = chosen
    current_order = neighbor
    tabu_list.append((i, j, iteration + tenure))

    # Mise à jour du meilleur
    if makespan < best_makespan:
        best_order = [gene[:] for gene in neighbor]
        best_makespan = makespan

    return current_order, best_order, best_makespan, tabu_list


class TabuAgent(Agent):
    """Agent utilisant la recherche tabou sur chromosomes."""

    def __init__(self, model, collaboratif=False, tenure=7, candidate_size=40):
        super().__init__(model)

        # Initialiser le chromosome courant et le meilleur
        self.current_order = create_random_chromosome(
            self.model.competence_matrix)
        self.best_order = [gene[:] for gene in self.current_order]

        # Calculer le makespan initial
        solution = decode_chromosome(
            self.best_order, self.model.competence_matrix)
        self.makespan = calculate_makespan(solution)

        # Paramètres de la recherche tabu
        self.tabu_list = []  # Liste de tuples (i, j, iteration_expiration)
        self.iteration = 1
        self.tenure = tenure
        self.candidate_size = candidate_size
        self.collaboratif = collaboratif

    def tabu_search_step(self):
        """Recherche tabu - une itération"""
        self.current_order, self.best_order, self.makespan, self.tabu_list = tabu_search_step(
            self.current_order,
            self.best_order,
            self.makespan,
            self.tabu_list,
            self.iteration,
            self.model.competence_matrix,
            tenure=self.tenure,
            candidate_size=self.candidate_size
        )
        self.iteration += 1

    def contact(self):
        """
        Si collaboratif, récupère le meilleur chromosome parmi tous les agents
        """
        if not self.collaboratif:
            return

        min_makespan = self.makespan
        best_agent = None

        for a in self.model.my_agents:
            if hasattr(a, 'makespan') and hasattr(a, 'best_order') and a.makespan < min_makespan:
                min_makespan = a.makespan
                best_agent = a

        if best_agent is not None:
            self.makespan = best_agent.makespan
            self.best_order = [gene[:] for gene in best_agent.best_order]
            self.current_order = [gene[:] for gene in best_agent.best_order]

    def step(self):
        """Exécute une étape de recherche tabou"""
        self.tabu_search_step()
        if self.collaboratif:
            self.contact()
