"""
Tests de l'algorithme de recherche Tabu avec différentes tailles de matrices.
"""
from src.algorithms.tabu import tabu_search, normalize_competence_matrix, schedule_to_ordered_operations
from src.data.generator import load_competence_matrix
from src.decoding.decoder import decode_chromosome
from src.evaluation.fitness import calculate_makespan
import json
import sys
import time
from pathlib import Path

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_tabu_on_matrix(matrix_file, tenures, max_iterations=500, candidate_size=50):
    """
    Teste l'algorithme Tabu sur une matrice donnée avec différentes tenures.

    Args:
        matrix_file: Nom du fichier de matrice de compétences
        tenures: Liste des tenures (durée tabu) à tester
        max_iterations: Nombre d'itérations max
        candidate_size: Taille de la liste de candidats

    Returns:
        Liste des résultats {tenure, cmax, iterations, time}
    """
    print(f"\n{'='*70}")
    print(f"Test sur: {matrix_file}")
    print(f"{'='*70}")

    # Charger la matrice
    C = load_competence_matrix(matrix_file)
    C_array = normalize_competence_matrix(C)

    num_patients = len(C)
    total_ops = sum(len(patient) for patient in C)

    print(f"Patients: {num_patients} | Operations totales: {total_ops}")
    print(f"Iterations max: {max_iterations} | Candidats: {candidate_size}")
    print()

    results = []

    for tenure in tenures:
        print(f"Tenure: {tenure:3d} | ", end='', flush=True)

        start_time = time.time()

        # Exécuter l'algorithme Tabu
        best_sched, best_cmax, iterations, history = tabu_search(
            C_array,
            max_iter=max_iterations,
            tenure=tenure,
            candidate_size=candidate_size,
            seed=42,
            verbose=False,
            track_history=True,
            max_no_improve=100
        )

        # Convertir en liste d'opérations et recalculer avec décodeur commun
        ordered_operations = schedule_to_ordered_operations(
            best_sched, C_array)
        solution = decode_chromosome(ordered_operations, C_array)
        cmax = calculate_makespan(solution)

        elapsed = time.time() - start_time

        result = {
            'matrix_file': matrix_file,
            'num_patients': num_patients,
            'total_operations': total_ops,
            'tenure': tenure,
            'max_iterations': max_iterations,
            'candidate_size': candidate_size,
            'cmax': cmax,
            'iterations_done': iterations,
            'time': elapsed
        }
        results.append(result)

        print(
            f"CMax: {cmax:3d} | Iterations: {iterations:4d} | Temps: {elapsed:6.2f}s")

    return results


def main():
    """Exécute tous les tests."""
    print("="*70)
    print(" "*15 + "TESTS RECHERCHE TABU")
    print("="*70)

    # Configurations de test
    matrix_files = [
        'competence_matrix_10.json',
        'competence_matrix_50.json',
        'competence_matrix_100.json'
    ]

    # Différentes tenures à tester
    tenure_configs = {
        'competence_matrix_10.json': [3, 5, 7, 10],
        'competence_matrix_50.json': [5, 7, 10, 15],
        'competence_matrix_100.json': [7, 10, 15, 20]
    }

    # Nombre d'itérations et taille de candidats adaptés à la taille
    iteration_configs = {
        'competence_matrix_10.json': {'max_iter': 300, 'candidate_size': 30},
        'competence_matrix_50.json': {'max_iter': 500, 'candidate_size': 50},
        'competence_matrix_100.json': {'max_iter': 600, 'candidate_size': 70}
    }

    all_results = []

    # Tester chaque matrice
    for matrix_file in matrix_files:
        try:
            tenures = tenure_configs[matrix_file]
            config = iteration_configs[matrix_file]
            results = test_tabu_on_matrix(
                matrix_file,
                tenures,
                max_iterations=config['max_iter'],
                candidate_size=config['candidate_size']
            )
            all_results.extend(results)
        except FileNotFoundError as e:
            print(f"\n[ERREUR] Fichier non trouve: {matrix_file}")
            print(f"Veuillez executer: python -m src.data.generator")
            continue

    # Sauvegarder les résultats
    if all_results:
        output_dir = Path(__file__).parent / 'results'
        output_dir.mkdir(exist_ok=True)

        output_file = output_dir / 'tabu_results.json'
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
                      f"(tenure={best['tenure']:3d}, "
                      f"temps={best['time']:.2f}s)")


if __name__ == "__main__":
    main()
