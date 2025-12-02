import random

from src.decoding.decoder import decode_chromosome
from src.evaluation.fitness import calculate_makespan

# ==========================
#   DONNÉES
# ==========================


# ==========================
#   CHROMOSOMES
# ==========================

def create_random_chromosome(competence_matrix):
    """Chromosome = permutation de toutes les opérations (patient, op_idx)."""
    chromosome = []
    nb_patient = len(competence_matrix)
    for p in range(nb_patient):
        for op_idx in range(len(competence_matrix[p])):
            chromosome.append([p, op_idx])
    random.shuffle(chromosome)
    return chromosome


def create_initial_population(population_size, competence_matrix):
    return [create_random_chromosome(competence_matrix) for _ in range(population_size)]


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
                      verbose=True):

    population = create_initial_population(population_size, competence_matrix)

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

    best_solution = decode_chromosome(best_chrom, competence_matrix)
    return best_chrom, best_solution, best_ms, best_fit


# ==========================
#   MAIN
# ==========================


if __name__ == "__main__":
    from src.data.instances import competence_matrix
    from src.visualization.display import plot_planning
    from src.utils.common import run_and_display

    def wrapper_genetic(comp_matrix, **params):
        """Wrapper pour adapter genetic_algorithm au format run_and_display."""
        return genetic_algorithm(comp_matrix, **params)

    run_and_display(
        algorithm_name="Algorithme Génétique",
        algorithm_func=wrapper_genetic,
        competence_matrix=competence_matrix,
        plot_func=plot_planning,
        calculate_makespan_func=calculate_makespan,
        population_size=20,
        generations=50,
        mutation_rate=0.2,
        verbose=True
    )
