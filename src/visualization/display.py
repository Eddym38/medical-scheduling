"""
Module de visualisation des solutions de planification.
"""
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches


def plot_planning(solution, title="Planning", save_path=None):
    """
    Affiche le planning :
    - lignes = compétences (skills)
    - colonnes = temps
    - couleur = patient
    - texte dans la case = numéro d'opération

    Args:
        solution: Matrice [skill][time] avec (patient, op_idx) ou None
        title: Titre du graphique
        save_path: Chemin pour sauvegarder l'image (optionnel)
    """
    if not solution or not solution[0]:
        print("Solution vide, rien à afficher.")
        return

    nb_skills = len(solution)
    # Prendre le max au cas où les compétences ont des longueurs différentes
    horizon = max(len(skill) for skill in solution)

    patient_set = set()
    for skill_index in range(nb_skills):
        for slot in solution[skill_index]:
            if slot is not None:
                patient_index, _ = slot
                patient_set.add(patient_index)

    patient_list = sorted(list(patient_set))
    patient_to_int = {patient_index: i + 1 for i,
                      patient_index in enumerate(patient_list)}

    color_data = []
    op_data = []

    for skill_index in range(nb_skills):
        color_row = []
        op_row = []
        for t in range(horizon):
            # Vérifier si l'index existe pour cette compétence
            if t < len(solution[skill_index]):
                slot = solution[skill_index][t]
                if slot is None:
                    color_row.append(0)
                    op_row.append(None)
                else:
                    patient_index, operation_index = slot
                    color_row.append(patient_to_int[patient_index])
                    op_row.append(operation_index + 1)
            else:
                # Compléter avec des cases vides si la compétence est plus courte
                color_row.append(0)
                op_row.append(None)
        color_data.append(color_row)
        op_data.append(op_row)

    nb_colors = len(patient_list) + 1
    base_cmap = plt.get_cmap("tab20")
    colors = [(1, 1, 1, 1)]
    for i in range(len(patient_list)):
        colors.append(base_cmap(i))
    cmap = ListedColormap(colors)

    fig, ax = plt.subplots(
        figsize=(max(6, horizon * 0.4), max(3, nb_skills * 0.6)))
    ax.imshow(color_data, cmap=cmap, aspect="auto", origin="upper")

    for skill_index in range(nb_skills):
        for t in range(horizon):
            op_number = op_data[skill_index][t]
            if op_number is not None:
                ax.text(
                    t,
                    skill_index,
                    str(op_number),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                )

    ax.set_xticks(range(horizon))
    ax.set_yticks(range(nb_skills))
    ax.set_xticklabels(range(horizon))
    ax.set_yticklabels([f"Skill {i+1}" for i in range(nb_skills)])
    ax.set_xlabel("Temps")
    ax.set_ylabel("Compétences")
    ax.set_title(title)

    ax.set_xticks([x - 0.5 for x in range(1, horizon)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, nb_skills)], minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.3)
    ax.tick_params(axis="both", which="both", length=0)

    legend_patches = []
    for patient_index in patient_list:
        color_index = patient_to_int[patient_index]
        patch_color = colors[color_index]
        legend_patches.append(
            mpatches.Patch(color=patch_color,
                           label=f"Patient P{patient_index + 1}")
        )

    if legend_patches:
        ax.legend(
            handles=legend_patches,
            title="Patients",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            borderaxespad=0.,
            fontsize=8,
        )

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        print(f"Planning sauvegardé dans : {save_path}")

    plt.show()
