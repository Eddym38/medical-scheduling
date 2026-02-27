"""
Point d'entree pour lancer le systeme multi-agents complet (version notebook en .py).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

# Permet l'execution via "python agent/sma_main.py"
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.generator import load_competence_matrix
from sma import CoordinateurSMA


def infer_nb_competences(competence_matrix: List[List[List[int]]]) -> int:
    if not competence_matrix or not competence_matrix[0]:
        return 0
    return len(competence_matrix[0][0])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simulation du systeme multi-agents urgences (Accueil + Identificateur + Ordonnanceur)."
    )
    parser.add_argument(
        "--matrix",
        default="competence_matrix_10.json",
        help="Nom du fichier JSON dans agent/data/ a charger.",
    )
    parser.add_argument(
        "--mode",
        default="pipeline",
        choices=["pipeline", "parallele"],
        help="Mode d'ordonnancement.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=150,
        help="Nombre d'iterations pour le mode parallele (ou info de run).",
    )
    parser.add_argument(
        "--steps-par-phase",
        type=int,
        default=50,
        help="Nombre d'iterations par phase en mode pipeline.",
    )
    parser.add_argument(
        "--nb-personnel",
        type=int,
        default=10,
        help="Taille de l'equipe generee par l'Identificateur.",
    )
    parser.add_argument(
        "--max-patients",
        type=int,
        default=0,
        help="Limiter le batch aux N premiers patients (0 = tous).",
    )
    parser.add_argument(
        "--absence-id",
        type=int,
        default=0,
        help="ID du personnel a rendre absent apres le premier planning (0 = aucune absence).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed aleatoire pour run reproductible.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Affichage minimal.",
    )
    parser.add_argument(
        "--show-registry",
        action="store_true",
        help="Afficher le registre du personnel avant ordonnancement.",
    )
    parser.add_argument(
        "--show-queue",
        action="store_true",
        help="Afficher la file d'attente avant ordonnancement.",
    )
    parser.add_argument(
        "--show-summary",
        action="store_true",
        help="Afficher un bilan final detaille.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Afficher le planning final sous forme de graphique.",
    )
    parser.add_argument(
        "--save-plot",
        default="",
        help="Chemin de sauvegarde de l'image du planning final (ex: planning_final.png).",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    competence_matrix = load_competence_matrix(args.matrix)
    if args.max_patients and args.max_patients > 0:
        competence_matrix = competence_matrix[: args.max_patients]

    nb_competences = infer_nb_competences(competence_matrix)
    sma = CoordinateurSMA(
        nb_competences=nb_competences,
        nb_personnel=args.nb_personnel,
        steps_par_phase=args.steps_par_phase,
        seed=args.seed,
    )

    sma.simuler_arrivees_batch(competence_matrix)

    if args.show_registry:
        sma.identificateur.afficher_registre()
    if args.show_queue:
        sma.accueil.afficher_file_attente()

    resultats = sma.lancer_ordonnancement(
        mode=args.mode,
        n_steps=args.steps,
        verbose=not args.quiet,
    )
    print(
        f"CMax initial: {resultats['makespan']} | "
        f"Patients: {len(competence_matrix)} | "
        f"Messages: {len(sma.messages_echanges)}"
    )

    if args.absence_id > 0:
        re_resultats = sma.simuler_absence(args.absence_id, verbose=not args.quiet)
        if re_resultats is not None:
            print(f"CMax apres absence: {re_resultats['makespan']}")
            resultats = re_resultats

    if args.show_summary:
        sma.afficher_bilan()

    if args.plot or args.save_plot:
        from src.visualization.display import plot_planning

        title = f"Planning final (CMax={resultats['makespan']})"
        save_path = args.save_plot if args.save_plot else None
        plot_planning(
            resultats["planning"],
            title=title,
            save_path=save_path,
            show=args.plot,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
