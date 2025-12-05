"""
Agent utilisant l'algorithme génétique.
"""
import random
import math
from mesa import Agent


class GeneticAgent(Agent):

    def __init__(self, model, inner_population_size=20, collaboratif=False, mutation_rate=0.2):
        """Initialisation de l'agent génétique"""
        super().__init__(model)

        # Créer une population initiale interne
        self.inner_population = create_initial_population(
            inner_population_size,
            self.model.competence_matrix
        )

        # Calculer les fitness et makespans de la population
        self.fitness_values, self.makespans = evaluate_population(
            self.inner_population,
            self.model.competence_matrix
        )

        # Trouver le meilleur chromosome initial
        best_index = min(range(len(self.inner_population)),
                         key=lambda i: self.makespans[i])
        self.best_order = [gene.copy()
                           for gene in self.inner_population[best_index]]
        self.makespan = self.makespans[best_index]

        # Paramètres de l'algorithme génétique
        self.population_size = inner_population_size
        self.mutation_rate = mutation_rate
        self.collaboratif = collaboratif

    def genetic(self):
        """Algorithme génétique - une seule génération"""

        # Conserver le meilleur élément
        new_population = [[gene.copy() for gene in self.best_order]]

        # Générer le reste de la population par croisement et mutation
        while len(new_population) < self.population_size:
            parent1 = roulette_selection_with_fitness(
                self.inner_population, self.fitness_values)
            parent2 = roulette_selection_with_fitness(
                self.inner_population, self.fitness_values)

            child = LOX_chromosomes(parent1, parent2)
            child = mutate_chromosome(child, self.mutation_rate)

            new_population.append(child)

        # Remplacer la population
        self.inner_population = new_population

        # Recalculer les fitness et makespans
        self.fitness_values, self.makespans = evaluate_population(
            self.inner_population,
            self.model.competence_matrix
        )

        # Mettre à jour le meilleur
        best_index = min(range(len(self.inner_population)),
                         key=lambda i: self.makespans[i])
        self.best_order = [gene.copy()
                           for gene in self.inner_population[best_index]]
        self.makespan = self.makespans[best_index]

    def contact(self):
        """Récupère la meilleure solution des autres agents et l'ajoute à la population"""

        # Trouver la meilleure solution parmi tous les agents
        best_external_agent = None
        best_external_makespan = self.makespan

        for a in self.model.agents:
            if a != self and a.makespan < best_external_makespan:
                best_external_makespan = a.makespan
                best_external_agent = a

        # Si on a trouvé une meilleure solution externe
        if best_external_agent is not None:
            # Trouver l'indice du pire élément de notre population
            worst_index = max(range(len(self.inner_population)),
                              key=lambda i: self.makespans[i])

            # Remplacer le pire par la meilleure solution externe
            self.inner_population[worst_index] = [
                gene.copy() for gene in best_external_agent.best_order]

            # Recalculer le makespan et fitness de cet élément
            solution = decode_chromosome(
                self.inner_population[worst_index], self.model.competence_matrix)
            self.makespans[worst_index] = calculate_makespan(solution)
            self.fitness_values[worst_index] = 1 / \
                (1 + self.makespans[worst_index])

            # Mettre à jour le meilleur si nécessaire
            if best_external_makespan < self.makespan:
                self.best_order = [gene.copy()
                                   for gene in best_external_agent.best_order]
                self.makespan = best_external_makespan

    def step(self):
        """Étape d'exécution de l'agent"""
        self.genetic()
        if self.collaboratif:
            self.contact()


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

# ==========================
#   ÉVALUATION POPULATION
# ==========================


def evaluate_population(population, competence_matrix):
    """Calcule fitness et makespan pour toute la population (1 décodage par chromosome)."""
    fitness_values = []
    makespans = []
    for chrom in population:
        solution = decode_chromosome(chrom, competence_matrix)
        ms = calculate_makespan(solution)
        makespans.append(ms)
        fitness_values.append(1 / (1 + ms))
    return fitness_values, makespans


# ==========================
#   SÉLECTION (ROULETTE) AVEC FITNESS
# ==========================

def roulette_selection_with_fitness(population, fitness_values):
    total_fit = sum(fitness_values)
    if total_fit == 0:
        return [gene.copy() for gene in random.choice(population)]

    y = random.random()
    cumulative = 0.0

    for chrom, fit_val in zip(population, fitness_values):
        cumulative += fit_val / total_fit
        if cumulative >= y:
            return [gene.copy() for gene in chrom]

    return [gene.copy() for gene in population[-1]]


# ==========================
#   CROISEMENT LOX SUR CHROMOSOMES
# ==========================

def LOX_chromosomes(parent1, parent2):
    """LOX appliqué directement sur les gènes (patient, op_idx)."""
    size = len(parent1)
    a, b = sorted(random.sample(range(size), 2))

    child = [None] * size
    child[a:b+1] = [gene.copy() for gene in parent1[a:b+1]]

    p2_idx = 0
    for i in range(size):
        if child[i] is None:
            while parent2[p2_idx] in child:
                p2_idx += 1
            child[i] = parent2[p2_idx].copy()

    return child


# ==========================
#   MUTATION (SWAP DE GÈNES)
# ==========================

def mutate_chromosome(chromosome, mutation_rate=0.2):
    if random.random() > mutation_rate:
        return chromosome
    child = [gene.copy() for gene in chromosome]
    i, j = random.sample(range(len(child)), 2)
    child[i], child[j] = child[j], child[i]
    return child


# ==========================
#   ALGORITHME GÉNÉTIQUE
# ==========================

def genetic_algorithm(competence_matrix,
                      population_size=20,
                      generations=50,
                      mutation_rate=0.2,
                      initial_solution=None,
                      verbose=True):
    """
    Algorithme génétique pour l'ordonnancement.

    Args:
        competence_matrix: Matrice de compétences
        population_size: Taille de la population
        generations: Nombre de générations
        mutation_rate: Taux de mutation
        initial_solution: Solution initiale optionnelle [[patient, op], ...]
        verbose: Affichage des informations

    Returns:
        Liste d'opérations ordonnées [[patient, op], ...]
    """
    # Créer la population initiale
    if initial_solution is not None:
        # Utiliser la solution initiale et générer le reste aléatoirement
        population = [[gene.copy() for gene in initial_solution]]
        population.extend(create_initial_population(
            population_size - 1, competence_matrix))
    else:
        population = create_initial_population(
            population_size, competence_matrix)

    fitness_values, makespans = evaluate_population(
        population, competence_matrix)
    best_index = min(range(len(population)), key=lambda i: makespans[i])
    best_chrom = [gene.copy() for gene in population[best_index]]
    best_ms = makespans[best_index]
    best_fit = fitness_values[best_index]

    if verbose:
        print(
            f"Génération   0 | best makespan = {best_ms} | fitness = {best_fit:.6f}")

    for g in range(1, generations + 1):

        new_population = [[gene.copy() for gene in best_chrom]]

        while len(new_population) < population_size:
            parent1 = roulette_selection_with_fitness(
                population, fitness_values)
            parent2 = roulette_selection_with_fitness(
                population, fitness_values)

            child = LOX_chromosomes(parent1, parent2)
            child = mutate_chromosome(child, mutation_rate)

            new_population.append(child)

        population = new_population
        fitness_values, makespans = evaluate_population(
            population, competence_matrix)

        current_best_index = min(
            range(len(population)), key=lambda i: makespans[i])
        current_best_ms = makespans[current_best_index]
        current_best_fit = fitness_values[current_best_index]

        if current_best_ms < best_ms:
            best_ms = current_best_ms
            best_fit = current_best_fit
            best_chrom = [gene.copy()
                          for gene in population[current_best_index]]

        if verbose and (g % 10 == 0 or g == generations):
            print(
                f"Génération {g:3d} | best makespan = {best_ms} | fitness = {best_fit:.6f}")

    return best_chrom


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


def calculate_population_best_makespan(population, competence_matrix):
    """Calcule le makespan du meilleur chromosome dans la population."""
    best_makespan = math.inf
    best_chromosome = None
    for chrom in population:
        solution = decode_chromosome(chrom, competence_matrix)
        ms = calculate_makespan(solution)
        if ms < best_makespan:
            best_makespan = ms
            best_chromosome = chrom
    return best_chromosome, best_makespan
