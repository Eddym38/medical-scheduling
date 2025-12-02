"""
Script principal pour exécuter tous les tests sur les trois algorithmes.
"""
import time
import subprocess
import sys
from pathlib import Path

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_test(test_script, name):
    """Exécute un script de test et affiche le résultat."""
    print("\n" + "="*80)
    print(f" "*30 + f"LANCEMENT: {name}")
    print("="*80 + "\n")

    start = time.time()

    try:
        result = subprocess.run(
            [sys.executable, test_script],
            cwd=Path(__file__).parent,
            check=True,
            capture_output=False,
            text=True
        )
        elapsed = time.time() - start
        print(f"\n[OK] {name} termine en {elapsed:.2f}s")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start
        print(f"\n[ERREUR] {name} a echoue apres {elapsed:.2f}s")
        return False


def main():
    """Exécute tous les tests."""
    print("="*80)
    print(" "*20 + "SUITE DE TESTS - ALGORITHMES D'ORDONNANCEMENT")
    print("="*80)

    # Vérifier que les matrices existent
    data_dir = Path(__file__).parent.parent / 'src' / 'data'
    required_files = [
        'competence_matrix_10.json',
        'competence_matrix_50.json',
        'competence_matrix_100.json'
    ]

    missing_files = [f for f in required_files if not (data_dir / f).exists()]

    if missing_files:
        print("\n[ATTENTION] Matrices de competences manquantes:")
        for f in missing_files:
            print(f"  - {f}")
        print("\nGeneration des matrices...")

        try:
            subprocess.run(
                [sys.executable, "-m", "src.data.generator"],
                cwd=Path(__file__).parent.parent,
                check=True
            )
            print("[OK] Matrices generees avec succes\n")
        except subprocess.CalledProcessError:
            print("[ERREUR] Impossible de generer les matrices")
            return

    # Scripts de test à exécuter
    tests = [
        ("test_genetic.py", "Tests Algorithme Genetique"),
        ("test_simulated_annealing.py", "Tests Recuit Simule"),
        ("test_tabu.py", "Tests Recherche Tabu")
    ]

    results = {}
    total_start = time.time()

    for script, name in tests:
        success = run_test(script, name)
        results[name] = success

    total_elapsed = time.time() - total_start

    # Résumé final
    print("\n" + "="*80)
    print(" "*30 + "RESUME FINAL")
    print("="*80)

    for name, success in results.items():
        status = "[OK]" if success else "[ERREUR]"
        print(f"{status} {name}")

    print(f"\nTemps total d'execution: {total_elapsed:.2f}s")
    print("="*80)

    # Afficher l'emplacement des résultats
    results_dir = Path(__file__).parent / 'results'
    if results_dir.exists():
        print("\nResultats disponibles dans:")
        for result_file in results_dir.glob("*.json"):
            print(f"  - {result_file.name}")


if __name__ == "__main__":
    main()
