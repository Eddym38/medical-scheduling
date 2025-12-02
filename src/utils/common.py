"""
Fonctions utilitaires communes à tous les algorithmes d'ordonnancement.
"""


def run_and_display(algorithm_name, algorithm_func, competence_matrix,
                    plot_func, calculate_makespan_func, **algo_params):
    """
    Exécute un algorithme d'ordonnancement et affiche les résultats.

    Args:
        algorithm_name: Nom de l'algorithme (pour l'affichage)
        algorithm_func: Fonction de l'algorithme à exécuter
        competence_matrix: Matrice de compétences
        plot_func: Fonction de visualisation
        calculate_makespan_func: Fonction de calcul du makespan
        **algo_params: Paramètres à passer à l'algorithme

    Returns:
        Résultats de l'algorithme
    """
    import time

    print(f"\n{'='*60}")
    print(f"    {algorithm_name.upper()}")
    print(f"{'='*60}\n")

    start_time = time.time()
    results = algorithm_func(competence_matrix, **algo_params)
    elapsed = time.time() - start_time

    # Extraction des résultats selon le format de retour
    if isinstance(results, tuple):
        solution = results[1] if len(results) > 1 else results[0]
        makespan = results[2] if len(
            results) > 2 else calculate_makespan_func(solution)
    else:
        solution = results
        makespan = calculate_makespan_func(solution)

    print(f"\n{'='*60}")
    print(f"    RÉSULTATS FINAUX")
    print(f"{'='*60}")
    print(f"Makespan (CMax) : {makespan}")
    print(f"Temps d'exécution : {elapsed:.2f} secondes")

    # Visualisation
    plot_func(solution, title=f"{algorithm_name} (CMax = {makespan})")

    return results
