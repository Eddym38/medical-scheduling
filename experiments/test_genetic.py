"""
Tests de l'algorithme génétique avec différentes tailles de matrices
et différents paramètres de population.
"""
from src.algorithms.genetic import genetic_algorithm
from src.data.generator import load_competence_matrix
from src.decoding.decoder import decode_chromosome
from src.evaluation.fitness import calculate_makespan
import json
import sys
import time
from pathlib import Path

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_genetic_on_matrix(matrix_file, pop_sizes, num_generations=100):
    """
    Teste l'algorithme génétique sur une matrice donnée avec différentes tailles de population.

    Args:
        matrix_file: Nom du fichier de matrice de compétences
        pop_sizes: Liste des tailles de population à tester
        num_generations: Nombre de générations

    Returns:
        Liste des résultats {pop_size, cmax, fitness, time, generations}
    """
    print(f"\n{'='*70}")
    print(f"Test sur: {matrix_file}")
    print(f"{'='*70}")

    # Charger la matrice
    C = load_competence_matrix(matrix_file)
    num_patients = len(C)
    total_ops = sum(len(patient) for patient in C)

    print(f"Patients: {num_patients} | Operations totales: {total_ops}")
    print(f"Generations: {num_generations}")
    print()

    results = []

    for pop_size in pop_sizes:
        print(f"Population: {pop_size:4d} | ", end='', flush=True)

        start_time = time.time()

        # Exécuter l'algorithme génétique
        best_chrom, best_solution, best_makespan, best_fitness = genetic_algorithm(
            competence_matrix=C,
            population_size=pop_size,
            generations=num_generations,
            mutation_rate=0.1,
            verbose=False
        )

        elapsed = time.time() - start_time

        result = {
            'matrix_file': matrix_file,
            'num_patients': num_patients,
            'total_operations': total_ops,
            'pop_size': pop_size,
            'num_generations': num_generations,
            'cmax': best_makespan,
            'fitness': best_fitness,
            'time': elapsed
        }
        results.append(result)

        print(
            f"CMax: {best_makespan:3d} | Fitness: {best_fitness:.6f} | Temps: {elapsed:6.2f}s")

    return results


def main():
    """Exécute tous les tests."""
    print("="*70)
    print(" "*15 + "TESTS ALGORITHME GENETIQUE")
    print("="*70)

    # Configurations de test
    matrix_files = [
        'competence_matrix_10.json',
        'competence_matrix_50.json',
        'competence_matrix_100.json'
    ]

    # Différentes tailles de population à tester
    pop_sizes_configs = {
        'competence_matrix_10.json': [20, 50, 100, 150],
        'competence_matrix_50.json': [50, 100, 200, 300],
        'competence_matrix_100.json': [100, 200, 300, 500]
    }

    num_generations = 100

    all_results = []

    # Tester chaque matrice
    for matrix_file in matrix_files:
        try:
            pop_sizes = pop_sizes_configs[matrix_file]
            results = test_genetic_on_matrix(
                matrix_file, pop_sizes, num_generations)
            all_results.extend(results)
        except FileNotFoundError as e:
            print(f"\n[ERREUR] Fichier non trouve: {matrix_file}")
            print(f"Veuillez executer: python -m src.data.generator")
            continue

    # Sauvegarder les résultats
    if all_results:
        output_dir = Path(__file__).parent / 'results'
        output_dir.mkdir(exist_ok=True)

        output_file = output_dir / 'genetic_results.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2)

        print(f"\n{'='*70}")
        print(f"Resultats sauvegardes dans: {output_file}")
        print(f"{'='*70}")

        # Résumé
        print("\nRESUME DES MEILLEURS RESULTATS:")
        print("-" * 70)

        for matrix_file in matrix_files:
            matrix_results = [
                r for r in all_results if r['matrix_file'] == matrix_file]
            if matrix_results:
                best = min(matrix_results, key=lambda x: x['cmax'])
                print(f"{matrix_file:30s} | "
                      f"Meilleur CMax: {best['cmax']:3d} "
                      f"(pop={best['pop_size']:4d}, "
                      f"temps={best['time']:.2f}s)")


if __name__ == "__main__":
    main()
