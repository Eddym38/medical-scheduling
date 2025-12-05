from pathlib import Path
import json
import random
from numpy import random as np_random
from typing import List
import matplotlib.pyplot as plt
import numpy as np


def create_patient_data(num_patients: int,
                        num_operations: int,
                        num_competences: int,
                        num_competence_needed_max: int,
                        length_competence_needed_max: int,
                        num_operations_needed_probability: float = 0.9,
                        num_competences_needed_probability: float = 0.3,
                        length_competence_needed_probability: float = 0.5) -> List[List[List[int]]]:
    """
    Génère une matrice de compétences aléatoire pour les patients.

    Args:
        num_patients: Nombre de patients
        num_operations: Nombre max d'opérations par patient
        num_competences: Nombre total de compétences disponibles
        num_competence_needed_max: Nombre max de compétences différentes par opération
        length_competence_needed_max: Quantité max pour chaque compétence
        num_operations_needed_probability: Probabilité pour distribution géométrique (nb opérations)
        num_competences_needed_probability: Probabilité pour distribution géométrique (nb compétences)
        length_competence_needed_probability: Probabilité pour distribution géométrique (quantité)

    Returns:
        Matrice de compétences [patient][operation][competence]

    Example:
        >>> matrix = create_patient_data(3, 5, 6, 3, 5)
        >>> # Patient 0: [[2,0,0,0,0,0], [1,1,0,0,0,0], ...]
    """
    competence_matrix = []

    for _ in range(num_patients):
        patient_ops = []

        # Nombre d'opérations pour ce patient (distribution géométrique)
        num_operations_needed = max(
            1, min(np_random.geometric(
                p=1-num_operations_needed_probability), num_operations)
        )

        for _ in range(num_operations_needed):
            # Initialiser le vecteur de compétences pour cette opération
            operation_competences = [0] * num_competences

            # Nombre de compétences différentes pour cette opération
            num_competences_needed = max(
                1, min(np_random.geometric(p=1-num_competences_needed_probability),
                       num_competence_needed_max)
            )

            # Sélectionner aléatoirement les compétences requises
            selected_competences = random.sample(
                range(num_competences), k=num_competences_needed)

            # Pour chaque compétence sélectionnée, déterminer la quantité
            for comp_idx in selected_competences:
                quantity = max(
                    1, min(np_random.geometric(p=1-length_competence_needed_probability),
                           length_competence_needed_max)
                )
                operation_competences[comp_idx] = quantity

            patient_ops.append(operation_competences)

        competence_matrix.append(patient_ops)

    return competence_matrix


def plot_competence_matrix(competence_matrix: List[List[List[int]]]):
    """
    Affiche la matrice de compétences sous forme de heatmap.

    Chaque ligne représente un patient avec toutes ses opérations côte à côte.
    Format: [Patient 0: Op0 | Op1 | Op2 | Op3]
            [Patient 1: Op0 | Op1 | Op2 | Op3]
    """
    if not competence_matrix or not competence_matrix[0]:
        print("Matrice vide!")
        return

    num_patients = len(competence_matrix)
    num_competences = len(competence_matrix[0][0])

    # Trouver le nombre max d'opérations (peut varier par patient)
    max_operations = max(len(patient) for patient in competence_matrix)

    # Créer une matrice 2D : chaque ligne = 1 patient, colonnes = opérations * compétences
    matrix_2d = np.zeros((num_patients, max_operations * num_competences))

    for p in range(num_patients):
        num_ops = len(competence_matrix[p])
        for o in range(num_ops):
            # Placer chaque opération dans sa zone (o * num_competences)
            start_col = o * num_competences
            end_col = start_col + num_competences
            matrix_2d[p, start_col:end_col] = competence_matrix[p][o]

    # Créer la figure
    fig, ax = plt.subplots(
        figsize=(max(12, max_operations * 2), max(6, num_patients * 0.5))
    )

    # Afficher la heatmap
    im = ax.imshow(matrix_2d, cmap='YlOrRd',
                   aspect='auto', interpolation='nearest')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Niveau de compétence requis', rotation=270, labelpad=20)

    # Axes X : Opérations et Compétences
    x_labels = []
    for o in range(max_operations):
        for c in range(num_competences):
            x_labels.append(f"Op{o+1}\nC{c+1}")
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, fontsize=8, rotation=45, ha='right')

    # Axes Y : Patients
    ax.set_yticks(range(num_patients))
    ax.set_yticklabels([f"Patient {p+1}" for p in range(num_patients)])

    # Séparateurs visuels entre opérations (lignes noires verticales)
    for o in range(1, max_operations):
        ax.axvline(x=o * num_competences - 0.5, color='black', linewidth=2)

    # Grille légère
    ax.set_xticks(
        [x - 0.5 for x in range(1, max_operations * num_competences)], minor=True
    )
    ax.set_yticks([y - 0.5 for y in range(1, num_patients)], minor=True)
    ax.grid(which='minor', color='gray',
            linestyle='-', linewidth=0.5, alpha=0.3)

    # Labels et titre
    ax.set_xlabel('Opérations → Compétences', fontsize=10, fontweight='bold')
    ax.set_ylabel('Patients', fontsize=10, fontweight='bold')
    ax.set_title(
        f'Matrice de Compétences ({num_patients} patients)\n'
        '(Chaque ligne = 1 patient | Séparations noires = changement d\'opération)',
        fontsize=12, fontweight='bold', pad=20
    )

    plt.tight_layout()
    plt.show()


def save_competence_matrix(competence_matrix: List[List[List[int]]],
                           filename: str = "competence_matrix.json"):
    """Sauvegarde la matrice de compétences en JSON."""
    filepath = Path(__file__).parent / filename
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(competence_matrix, f, indent=2)
    print(f"[OK] Matrice sauvegardee dans: {filepath}")


def load_competence_matrix(filename: str = "competence_matrix.json") -> List[List[List[int]]]:
    """Charge une matrice de compétences depuis un fichier JSON."""
    filepath = Path(__file__).parent / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Fichier introuvable: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        competence_matrix = json.load(f)
    print(f"[OK] Matrice chargee depuis: {filepath}")
    return competence_matrix


# Test si exécuté directement
if __name__ == "__main__":
    print("="*60)
    print("GENERATION DES MATRICES DE TEST")
    print("="*60)

    # Configurations des différentes tailles
    configs = [
        {
            'name': 'small',
            'num_patients': 10,
            'num_operations': 3,
            'num_competences': 4,
            'num_competence_needed_max': 2,
            'length_competence_needed_max': 2,
            'filename': 'competence_matrix_10.json'
        },
        {
            'name': 'medium',
            'num_patients': 50,
            'num_operations': 5,
            'num_competences': 6,
            'num_competence_needed_max': 3,
            'length_competence_needed_max': 3,
            'filename': 'competence_matrix_50.json'
        },
        {
            'name': 'large',
            'num_patients': 100,
            'num_operations': 6,
            'num_competences': 8,
            'num_competence_needed_max': 4,
            'length_competence_needed_max': 4,
            'filename': 'competence_matrix_100.json'
        }
    ]

    for config in configs:
        print(f"\n{'='*60}")
        print(
            f"Generation matrice {config['name'].upper()} ({config['num_patients']} patients)...")

        matrix = create_patient_data(
            num_patients=config['num_patients'],
            num_operations=config['num_operations'],
            num_competences=config['num_competences'],
            num_competence_needed_max=config['num_competence_needed_max'],
            length_competence_needed_max=config['length_competence_needed_max']
        )

        save_competence_matrix(matrix, config['filename'])

        # Aperçu
        total_ops = sum(len(patient) for patient in matrix)
        print(f"  - Patients: {len(matrix)}")
        print(f"  - Operations totales: {total_ops}")
        print(f"  - Competences: {config['num_competences']}")

    print(f"\n{'='*60}")
    print("GENERATION TERMINEE")
    print("="*60)
    print("\nFichiers generes:")
    for config in configs:
        print(f"  - {config['filename']} ({config['num_patients']} patients)")

    # Sauvegarder aussi la matrice par défaut (20 patients)
    print(f"\n{'='*60}")
    print("Generation matrice DEFAULT (20 patients)...")
    default_matrix = create_patient_data(
        num_patients=20,
        num_operations=5,
        num_competences=6,
        num_competence_needed_max=3,
        length_competence_needed_max=3
    )
    save_competence_matrix(default_matrix, "competence_matrix.json")
    print("  - competence_matrix.json (20 patients)")
