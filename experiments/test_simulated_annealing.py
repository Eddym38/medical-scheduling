"""
Tests de l'algorithme de recuit simulé avec différentes tailles de matrices.
"""
from src.algorithms.simulated_annealing import algo_rs
from src.data.generator import load_competence_matrix
from src.decoding.decoder import decode_chromosome
from src.evaluation.fitness import calculate_makespan
import json
import sys
import time
from pathlib import Path

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_simulated_annealing_on_matrix(matrix_file, temperatures, max_iterations=1000):
    """
    Teste l'algorithme de recuit simulé sur une matrice donnée avec différentes températures.

    Args:
        matrix_file: Nom du fichier de matrice de compétences
        temperatures: Liste des températures initiales à tester
        max_iterations: Nombre d'itérations max

    Returns:
        Liste des résultats {temperature, cmax, time}
    """
    print(f"\n{'='*70}")
    print(f"Test sur: {matrix_file}")
    print(f"{'='*70}")

    # Charger la matrice
    C = load_competence_matrix(matrix_file)
    num_patients = len(C)
    total_ops = sum(len(patient) for patient in C)

    print(f"Patients: {num_patients} | Operations totales: {total_ops}")
    print(f"Iterations max: {max_iterations}")
    print()

    results = []

    for temp in temperatures:
        print(f"Temperature: {temp:6.1f} | ", end='', flush=True)

        start_time = time.time()

        # Exécuter l'algorithme de recuit simulé
        best_ordre = algo_rs(
            competence_matrix=C,
            t0=temp,
            alpha=0.95,
            nbiter_cycle=max_iterations
        )

        # Décoder et calculer le makespan
        solution = decode_chromosome(best_ordre, C)
        cmax = calculate_makespan(solution)

        elapsed = time.time() - start_time

        result = {
            'matrix_file': matrix_file,
            'num_patients': num_patients,
            'total_operations': total_ops,
            'temperature': temp,
            'max_iterations': max_iterations,
            'cmax': cmax,
            'time': elapsed
        }
        results.append(result)

        print(f"CMax: {cmax:3d} | Temps: {elapsed:6.2f}s")

    return results


def main():
    """Exécute tous les tests."""
    print("="*70)
    print(" "*15 + "TESTS RECUIT SIMULE")
    print("="*70)

    # Configurations de test
    matrix_files = [
        'competence_matrix_10.json',
        'competence_matrix_50.json',
        'competence_matrix_100.json'
    ]

    # Différentes températures à tester
    temp_configs = {
        'competence_matrix_10.json': [50, 100, 200, 500],
        'competence_matrix_50.json': [100, 200, 500, 1000],
        'competence_matrix_100.json': [200, 500, 1000, 2000]
    }

    # Nombre d'itérations adaptés à la taille
    iterations_configs = {
        'competence_matrix_10.json': 500,
        'competence_matrix_50.json': 1000,
        'competence_matrix_100.json': 2000
    }

    all_results = []

    # Tester chaque matrice
    for matrix_file in matrix_files:
        try:
            temperatures = temp_configs[matrix_file]
            max_iterations = iterations_configs[matrix_file]
            results = test_simulated_annealing_on_matrix(
                matrix_file, temperatures, max_iterations)
            all_results.extend(results)
        except FileNotFoundError as e:
            print(f"\n[ERREUR] Fichier non trouve: {matrix_file}")
            print(f"Veuillez executer: python -m src.data.generator")
            continue

    # Sauvegarder les résultats
    if all_results:
        output_dir = Path(__file__).parent / 'results'
        output_dir.mkdir(exist_ok=True)

        output_file = output_dir / 'simulated_annealing_results.json'
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
                      f"(temp={best['temperature']:6.1f}, "
                      f"temps={best['time']:.2f}s)")


if __name__ == "__main__":
    main()
