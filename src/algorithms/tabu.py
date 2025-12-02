import numpy as np
import random
from typing import Dict, List, Tuple


# ----------------------- Utilitaires de conversion -----------------------
def normalize_competence_matrix(competence_matrix: List[List[List[int]]]) -> np.ndarray:
    """
    Convertit une matrice de compétences irrégulière en tableau numpy régulier.
    Ajoute des opérations vides (que des 0) pour les patients avec moins d'opérations.

    Args:
        competence_matrix: Liste [patient][operation][skill] de tailles variables

    Returns:
        Tableau numpy de forme (num_patients, max_operations, num_skills)
    """
    num_patients = len(competence_matrix)
    max_operations = max(len(patient_ops) for patient_ops in competence_matrix)
    num_skills = len(competence_matrix[0][0])

    # Initialiser avec des zéros
    normalized = np.zeros(
        (num_patients, max_operations, num_skills), dtype=int)

    # Remplir avec les données réelles
    for p in range(num_patients):
        for o in range(len(competence_matrix[p])):
            for s in range(num_skills):
                normalized[p, o, s] = competence_matrix[p][o][s]

    return normalized


# ----------------------- Construction des tâches unitaires -----------------------
def build_tasks_by_comp(C: np.ndarray) -> Dict[int, List[Tuple[str, int, int, int]]]:
    """
    Renvoie pour chaque compétence k la liste des tâches unitaires sous forme :
      (task_id, patient p, operation o, comp k)
    Ordre initial simple : tri par patient puis opération.
    """
    P, O, K = C.shape
    by_comp: Dict[int, List[Tuple[str, int, int, int]]] = {
        k: [] for k in range(K)}
    for p in range(P):
        for o in range(O):
            for k in range(K):
                cnt = int(C[p, o, k])
                for u in range(cnt):
                    tid = f"p{p}_o{o}_k{k}_u{u}"
                    by_comp[k].append((tid, p, o, k))
    return by_comp


def parse_tid(tid: str):
    # "p{p}_o{o}_k{k}_u{u}"
    parts = tid.split("_")
    p = int(parts[0][1:])
    o = int(parts[1][1:])
    k = int(parts[2][1:])
    u = int(parts[3][1:])
    return p, o, k, u

# ----------------------- Simulation (évaluation + journal) -----------------------


def evaluate_with_logs(schedule: Dict[int, List[Tuple[str, int, int, int]]],
                       C: np.ndarray):
    """
    Simule un planning donné par 'schedule' (ordre prioritaire sur chaque compétence).
    Règles :
      - capacité 1 par compétence,
      - 1 tâche unitaire = 1 tick,
      - une tâche (p,o) n'est éligible que si l'opération o de p est la prochaine à faire.
    Retour : (cmax_réel, logs) avec logs = liste de dicts
             (comp, patient, op, start_tick, end_tick, task_id)
    Le cmax_réel est le tick de fin de la dernière tâche (pas le temps total)
    """
    P, O, K = C.shape
    queues = {k: lst[:] for k, lst in schedule.items()}      # copies (listes)
    running: Dict[int, Tuple[str, int]] = {k: None for k in range(K)}
    remaining = {(p, o): int(C[p, o].sum())
                 for p in range(P) for o in range(O)}

    # prochaine opération à traiter par patient
    next_op = {}
    for p in range(P):
        o = 0
        while o < O and remaining[(p, o)] == 0:
            o += 1
        next_op[p] = o  # = O s'il n'y a plus rien

    logs = []
    t = 0                   # en ticks
    total_tasks = sum(remaining.values())
    done = 0

    while done < total_tasks:
        # Démarrer ce qui est éligible à l'instant t
        for k in range(K):
            if running[k] is not None:
                continue
            q = queues[k]
            chosen = None
            for idx, (tid, p, o, _k) in enumerate(q):
                if next_op[p] == o:      # éligible
                    chosen = idx
                    break
            if chosen is not None:
                tid, p, o, _k = q[chosen]
                del q[chosen]
                running[k] = (tid, t + 1)          # finit à t+1
                logs.append({"comp": k, "patient": p, "op": o,
                             "task_id": tid, "start_tick": t, "end_tick": t + 1})

        # Sécurité : si rien ne tourne alors qu'il reste des tâches => incohérence dans l'ordonnancement
        if all(r is None for r in running.values()):
            raise RuntimeError(
                "Blocage : aucune tâche en cours ni éligible. Vérifie les données.")

        # Avancer d'1 tick
        t += 1

        # Terminer ce qui finit maintenant
        for k in range(K):
            r = running[k]
            if r is None:
                continue
            tid, end_tick = r
            if end_tick == t:
                running[k] = None
                p, o, _k, _u = parse_tid(tid)
                remaining[(p, o)] -= 1
                done += 1
                # Si toute l'opération (p,o) est finie, débloquer la suivante non nulle
                if remaining[(p, o)] == 0 and next_op[p] == o:
                    o2 = o + 1
                    while o2 < O and remaining[(p, o2)] == 0:
                        o2 += 1
                    next_op[p] = o2

    # Calculer le vrai makespan (dernière tâche terminée)
    if not logs:
        return 0, logs
    cmax_reel = max(log["end_tick"] for log in logs)
    return cmax_reel, logs


def eval_cmax(schedule: Dict[int, List[Tuple[str, int, int, int]]],
              C: np.ndarray) -> int:
    cmax, _ = evaluate_with_logs(schedule, C)
    return cmax

# ----------------------- Mouvement : insertion -----------------------


def apply_insertion(schedule: Dict[int, List[Tuple[str, int, int, int]]],
                    k: int, i: int, j: int) -> Dict[int, List[Tuple[str, int, int, int]]]:
    """ Insère l'élément i à la position j dans la file de la compétence k. """
    new = {kk: lst[:] for kk, lst in schedule.items()}
    if len(new[k]) < 2 or i == j:
        return new
    i = max(0, min(i, len(new[k]) - 1))
    j = max(0, min(j, len(new[k])))
    elem = new[k].pop(i)
    new[k].insert(j, elem)
    return new

# ----------------------- Tabu Search -----------------------


def tabu_search(C: np.ndarray,
                max_iter: int = 250,
                tenure: int = 7,
                candidate_size: int = 40,
                seed: int = 0,
                verbose: bool = False,
                track_history: bool = False,
                max_no_improve: int = 50):
    """
    Tabu search :
      - renvoie le meilleur planning, son Cmax (en ticks),
      - le nombre d'itérations réellement effectuées,
      - et éventuellement l'historique du Cmax par itération.
    Arrêt anticipé si aucune amélioration pendant max_no_improve itérations.
    """
    random.seed(seed)
    P, O, K = C.shape

    current = build_tasks_by_comp(C)   # ordre patient->op par ressource
    best = current
    best_val = eval_cmax(best, C)

    tabu: Dict[Tuple[int, str, int], int] = {}

    iterations_done = 0
    history = [] if track_history else None
    no_improve = 0

    for it in range(1, max_iter + 1):
        iterations_done = it
        candidates = []

        # Générer quelques insertions aléatoires
        for _ in range(candidate_size):
            k = random.randrange(K)
            if len(current[k]) < 2:
                continue
            i = random.randrange(len(current[k]))
            j = random.randrange(len(current[k]))
            if i == j:
                continue
            moved_id = current[k][i][0]
            attr = (k, moved_id, j)
            cand = apply_insertion(current, k, i, j)
            val = eval_cmax(cand, C)
            candidates.append((val, cand, attr))

        if not candidates:
            if verbose:
                print(f"Arrêt prématuré à l'itération {it} (aucun candidat).")
            break

        candidates.sort(key=lambda x: x[0])
        chosen = None
        for val, cand, attr in candidates:
            is_tabu = (attr in tabu) and (tabu[attr] > it)
            if (not is_tabu) or (val < best_val):   # aspiration
                chosen = (val, cand, attr)
                break
        if chosen is None:
            chosen = candidates[0]

        val, cand, attr = chosen
        current = cand
        tabu[attr] = it + tenure

        improved = False
        if val < best_val:
            best, best_val = cand, val
            improved = True
            no_improve = 0
        else:
            no_improve += 1

        if verbose and (it == 1 or it % 10 == 0 or improved):
            msg = f"Iteration {it:4d} | Cmax = {best_val}"
            if improved:
                msg += "  <- amelioration"
            print(msg)

        if track_history:
            history.append((it, val, best_val))

        if max_no_improve is not None and no_improve >= max_no_improve:
            if verbose:
                print(
                    f"Arrêt anticipé : aucun progrès depuis {max_no_improve} itérations (it={it}).")
            break

    return best, best_val, iterations_done, history


def schedule_to_ordered_operations(schedule: Dict[int, List[Tuple[str, int, int, int]]],
                                   C: np.ndarray) -> List[List[int]]:
    """
    Convertit un schedule Tabu en liste d'opérations ordonnées [[patient, op], ...].
    L'ordre est déterminé par la simulation de l'exécution.
    """
    _, logs = evaluate_with_logs(schedule, C)

    # Trier les logs par temps de fin pour obtenir l'ordre d'exécution
    sorted_logs = sorted(logs, key=lambda x: (
        x["end_tick"], x["comp"], x["patient"], x["op"]))

    # Extraire les opérations uniques dans l'ordre
    operations_seen = set()
    ordered_operations = []

    for log in sorted_logs:
        op_key = (log["patient"], log["op"])
        if op_key not in operations_seen:
            operations_seen.add(op_key)
            ordered_operations.append([log["patient"], log["op"]])

    return ordered_operations


def convert_schedule_to_planning(schedule: Dict[int, List[Tuple[str, int, int, int]]],
                                 C: np.ndarray):
    """
    Convertit le format schedule (dict de listes de tâches) vers le format
    attendu par plot_planning : liste de listes [patient_id, op_id] ou None

    Returns:
        tuple: (solution, cmax_réel)
    """
    nb_skills = len(schedule)

    # Simuler l'exécution pour obtenir les logs
    _, logs = evaluate_with_logs(schedule, C)

    # Trouver le temps réel de fin (dernière tâche terminée)
    if not logs:
        return [[] for _ in range(nb_skills)], 0

    max_end_tick = max(log["end_tick"] for log in logs)

    # Initialiser avec None
    solution = [[None for _ in range(max_end_tick)] for _ in range(nb_skills)]

    # Remplir le planning à partir des logs
    for log in logs:
        comp = log["comp"]
        patient = log["patient"]
        op = log["op"]
        start = log["start_tick"]
        end = log["end_tick"]

        for tick in range(start, end):
            if tick < max_end_tick:
                solution[comp][tick] = [patient, op]

    return solution, max_end_tick


if __name__ == "__main__":
    from src.data.instances import competence_matrix
    from src.visualization.display import plot_planning
    from src.decoding.decoder import decode_chromosome
    from src.evaluation.fitness import calculate_makespan

    # Paramètres
    MAX_ITER = 600
    TENURE = 7
    CANDIDATE_SIZE = 50
    SEED = 2
    MAX_NO_IMPROVE = 80

    print("="*60)
    print("    ALGORITHME DE RECHERCHE TABU")
    print("="*60)

    # Conversion de la liste irrégulière en numpy array régulier
    C_array = normalize_competence_matrix(competence_matrix)

    print("\nDébut de l'algorithme de Recherche Tabu...\n")

    # Exécution de l'algorithme
    import time
    start_time = time.time()

    best_sched, best_cmax, iterations, history = tabu_search(
        C_array,
        max_iter=MAX_ITER,
        tenure=TENURE,
        candidate_size=CANDIDATE_SIZE,
        seed=SEED,
        verbose=True,
        track_history=True,
        max_no_improve=MAX_NO_IMPROVE
    )

    elapsed = time.time() - start_time

    # Convertir le schedule en liste d'opérations ordonnées
    ordered_operations = schedule_to_ordered_operations(best_sched, C_array)

    # Utiliser le décodeur commun
    solution_optimale = decode_chromosome(ordered_operations, C_array)
    cmax_reel = calculate_makespan(solution_optimale)

    # Résultats
    print("\n" + "="*60)
    print("    RÉSULTATS FINAUX")
    print("="*60)
    print(f"Makespan (CMax) : {cmax_reel}")
    print(f"Temps d'exécution : {elapsed:.2f} secondes")

    plot_planning(
        solution_optimale,
        title=f"Recherche Tabu (CMax = {cmax_reel})"
    )
