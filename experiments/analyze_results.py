"""
Analyse et comparaison des résultats des tests.
"""
import json
from pathlib import Path
import sys


def load_results(filename):
    """Charge les résultats depuis un fichier JSON."""
    filepath = Path(__file__).parent / 'results' / filename
    if not filepath.exists():
        return None

    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_genetic_results(results):
    """Analyse les résultats de l'algorithme génétique."""
    print("\n" + "="*80)
    print(" "*25 + "ALGORITHME GENETIQUE")
    print("="*80)

    # Grouper par matrice
    by_matrix = {}
    for r in results:
        matrix = r['matrix_file']
        if matrix not in by_matrix:
            by_matrix[matrix] = []
        by_matrix[matrix].append(r)

    for matrix, matrix_results in sorted(by_matrix.items()):
        print(f"\n{matrix} ({matrix_results[0]['num_patients']} patients):")
        print("-" * 80)
        print(f"{'Population':>12} | {'CMax':>6} | {'Fitness':>10} | {'Temps (s)':>10}")
        print("-" * 80)

        for r in sorted(matrix_results, key=lambda x: x['pop_size']):
            print(
                f"{r['pop_size']:12d} | {r['cmax']:6d} | {r['fitness']:10.6f} | {r['time']:10.2f}")

        # Meilleur résultat
        best = min(matrix_results, key=lambda x: x['cmax'])
        print(
            f"\nMeilleur: Population={best['pop_size']}, CMax={best['cmax']}, Temps={best['time']:.2f}s")


def analyze_simulated_annealing_results(results):
    """Analyse les résultats du recuit simulé."""
    print("\n" + "="*80)
    print(" "*25 + "RECUIT SIMULE")
    print("="*80)

    # Grouper par matrice
    by_matrix = {}
    for r in results:
        matrix = r['matrix_file']
        if matrix not in by_matrix:
            by_matrix[matrix] = []
        by_matrix[matrix].append(r)

    for matrix, matrix_results in sorted(by_matrix.items()):
        print(f"\n{matrix} ({matrix_results[0]['num_patients']} patients):")
        print("-" * 80)
        print(f"{'Temperature':>12} | {'CMax':>6} | {'Temps (s)':>10}")
        print("-" * 80)

        for r in sorted(matrix_results, key=lambda x: x['temperature']):
            print(f"{r['temperature']:12.1f} | {r['cmax']:6d} | {r['time']:10.2f}")

        # Meilleur résultat
        best = min(matrix_results, key=lambda x: x['cmax'])
        print(
            f"\nMeilleur: Temperature={best['temperature']:.1f}, CMax={best['cmax']}, Temps={best['time']:.2f}s")


def analyze_tabu_results(results):
    """Analyse les résultats de la recherche Tabu."""
    print("\n" + "="*80)
    print(" "*25 + "RECHERCHE TABU")
    print("="*80)

    # Grouper par matrice
    by_matrix = {}
    for r in results:
        matrix = r['matrix_file']
        if matrix not in by_matrix:
            by_matrix[matrix] = []
        by_matrix[matrix].append(r)

    for matrix, matrix_results in sorted(by_matrix.items()):
        print(f"\n{matrix} ({matrix_results[0]['num_patients']} patients):")
        print("-" * 80)
        print(f"{'Tenure':>12} | {'CMax':>6} | {'Iterations':>11} | {'Temps (s)':>10}")
        print("-" * 80)

        for r in sorted(matrix_results, key=lambda x: x['tenure']):
            print(
                f"{r['tenure']:12d} | {r['cmax']:6d} | {r['iterations_done']:11d} | {r['time']:10.2f}")

        # Meilleur résultat
        best = min(matrix_results, key=lambda x: x['cmax'])
        print(
            f"\nMeilleur: Tenure={best['tenure']}, CMax={best['cmax']}, Temps={best['time']:.2f}s")


def compare_algorithms():
    """Compare les trois algorithmes."""
    print("\n" + "="*80)
    print(" "*25 + "COMPARAISON DES ALGORITHMES")
    print("="*80)

    # Charger tous les résultats
    genetic_results = load_results('genetic_results.json')
    sa_results = load_results('simulated_annealing_results.json')
    tabu_results = load_results('tabu_results.json')

    if not all([genetic_results, sa_results, tabu_results]):
        print("\n[ERREUR] Tous les fichiers de resultats ne sont pas disponibles.")
        return

    # Grouper par taille de matrice
    matrices = sorted(set([r['matrix_file'] for r in genetic_results]))

    print(
        f"\n{'Matrice':30s} | {'Algorithme':20s} | {'CMax':>6s} | {'Temps (s)':>10s}")
    print("-" * 80)

    for matrix in matrices:
        # Meilleurs résultats pour chaque algo
        gen_best = min([r for r in genetic_results if r['matrix_file'] == matrix],
                       key=lambda x: x['cmax'])
        sa_best = min([r for r in sa_results if r['matrix_file'] == matrix],
                      key=lambda x: x['cmax'])
        tabu_best = min([r for r in tabu_results if r['matrix_file'] == matrix],
                        key=lambda x: x['cmax'])

        num_patients = gen_best['num_patients']

        print(
            f"{matrix:30s} | {'Genetique':20s} | {gen_best['cmax']:6d} | {gen_best['time']:10.2f}")
        print(f"{' '*30s} | {'Recuit Simule':20s} | {sa_best['cmax']:6d} | {sa_best['time']:10.2f}")
        print(f"{' '*30s} | {'Tabu':20s} | {tabu_best['cmax']:6d} | {tabu_best['time']:10.2f}")

        # Identifier le meilleur
        all_results = [
            ('Genetique', gen_best['cmax'], gen_best['time']),
            ('Recuit Simule', sa_best['cmax'], sa_best['time']),
            ('Tabu', tabu_best['cmax'], tabu_best['time'])
        ]
        winner = min(all_results, key=lambda x: x[1])
        print(f"{' '*30s} | ** Meilleur: {winner[0]} (CMax={winner[1]}) **")
        print("-" * 80)


def main():
    """Fonction principale."""
    print("="*80)
    print(" "*20 + "ANALYSE DES RESULTATS DE TESTS")
    print("="*80)

    results_dir = Path(__file__).parent / 'results'
    if not results_dir.exists():
        print("\n[ERREUR] Repertoire 'results' introuvable.")
        print("Veuillez d'abord executer: python run_all_tests.py")
        return

    # Charger et analyser chaque algorithme
    genetic_results = load_results('genetic_results.json')
    if genetic_results:
        analyze_genetic_results(genetic_results)

    sa_results = load_results('simulated_annealing_results.json')
    if sa_results:
        analyze_simulated_annealing_results(sa_results)

    tabu_results = load_results('tabu_results.json')
    if tabu_results:
        analyze_tabu_results(tabu_results)

    # Comparaison globale
    compare_algorithms()

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
