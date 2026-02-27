"""
Implementation executable du systeme multi-agents decrite dans le notebook.

Ce module ajoute une couche "metier urgences" autour du moteur
d'ordonnancement (recuit simule, tabou, genetique).
"""

from __future__ import annotations

import math
import random
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Permet l'import de src/* meme quand execute via "python agent/sma_main.py"
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.decoding.decoder import decode_chromosome
from src.evaluation.fitness import calculate_makespan
from src.utils.population import create_random_solution

CompetenceMatrix = List[List[List[int]]]
Chromosome = List[List[int]]


def evaluate_solution(operations: Chromosome, competence_matrix: CompetenceMatrix) -> int:
    """Evalue une solution (chromosome) et retourne son makespan."""
    solution = decode_chromosome(operations, competence_matrix)
    return calculate_makespan(solution)


class Urgence(Enum):
    """Niveaux de tri des urgences."""

    CRITIQUE = 1
    TRES_URGENT = 2
    URGENT = 3
    MOINS_URGENT = 4
    NON_URGENT = 5


class MessageType(Enum):
    """Types de messages echanges entre agents."""

    NOUVEAU_PATIENT = "nouveau_patient"
    DEMANDE_RESSOURCES = "demande_ressources"
    REPONSE_RESSOURCES = "reponse_ressources"
    LANCER_ORDONNANCEMENT = "lancer_ordonnancement"
    PLANNING_GENERE = "planning_genere"
    ABSENCE_PERSONNEL = "absence_personnel"
    RE_ORDONNANCEMENT = "re_ordonnancement"


class StatutPersonnel(Enum):
    """Statut d'un membre du personnel."""

    DISPONIBLE = "disponible"
    OCCUPE = "occupe"
    ABSENT = "absent"
    EN_PAUSE = "en_pause"


@dataclass
class Patient:
    """Patient arrivant au service d'urgences."""

    id: int
    nom: str
    urgence: Urgence
    heure_arrivee: float
    operations: List[List[int]] = field(default_factory=list)
    competences_requises: Set[int] = field(default_factory=set)
    statut: str = "en_attente"


@dataclass
class Personnel:
    """Membre du personnel hospitalier."""

    id: int
    nom: str
    competences: List[int]
    statut: StatutPersonnel = StatutPersonnel.DISPONIBLE
    charge_travail: float = 0.0


@dataclass
class Message:
    """Message echange entre agents."""

    type: MessageType
    emetteur: str
    destinataire: str
    contenu: Any
    timestamp: float = 0.0
    priorite: int = 0

    def __repr__(self) -> str:
        return (
            f"[{self.type.value}] {self.emetteur} -> "
            f"{self.destinataire} (t={self.timestamp:.1f})"
        )


class SousAgentRecuitSimule:
    """Sous-agent recuit simule pour l'exploration initiale."""

    def __init__(
        self,
        competence_matrix: CompetenceMatrix,
        temp_init: float = 1000.0,
        cooling_rate: float = 0.95,
    ) -> None:
        self.competence_matrix = competence_matrix
        self.temperature = temp_init
        self.temp_init = temp_init
        self.cooling_rate = cooling_rate
        self.nom = "RecuitSimule"

        self.best_order: Chromosome = create_random_solution(competence_matrix)
        self.makespan: int = evaluate_solution(self.best_order, competence_matrix)
        self.historique_makespan: List[int] = [self.makespan]

    @staticmethod
    def _verifier_precedence(ordre: Chromosome) -> bool:
        dernieres_ops: Dict[int, int] = {}
        for patient_id, op_id in ordre:
            if patient_id in dernieres_ops and op_id <= dernieres_ops[patient_id]:
                return False
            dernieres_ops[patient_id] = op_id
        return True

    def _generate_voisin(self, ordre: Chromosome) -> Chromosome:
        if len(ordre) < 2:
            return [gene[:] for gene in ordre]

        for _ in range(100):
            voisin = [gene[:] for gene in ordre]
            i, j = random.sample(range(len(voisin)), 2)

            if voisin[i][0] != voisin[j][0]:
                voisin[i], voisin[j] = voisin[j], voisin[i]
                if self._verifier_precedence(voisin):
                    return voisin

        return [gene[:] for gene in ordre]

    def step(self) -> None:
        voisin = self._generate_voisin(self.best_order)
        makespan_voisin = evaluate_solution(voisin, self.competence_matrix)

        delta = makespan_voisin - self.makespan
        if delta < 0:
            self.best_order = voisin
            self.makespan = makespan_voisin
        elif self.temperature > 1e-8:
            proba = math.exp(-delta / self.temperature)
            if random.random() < proba:
                self.best_order = voisin
                self.makespan = makespan_voisin

        self.temperature *= self.cooling_rate
        self.historique_makespan.append(self.makespan)

    def reset(self, solution: Optional[Chromosome] = None) -> None:
        if solution:
            self.best_order = [gene[:] for gene in solution]
            self.makespan = evaluate_solution(self.best_order, self.competence_matrix)
        self.temperature = self.temp_init


class SousAgentTabou:
    """Sous-agent recherche tabou pour l'intensification."""

    def __init__(
        self,
        competence_matrix: CompetenceMatrix,
        tenure: int = 7,
        candidate_size: int = 40,
    ) -> None:
        self.competence_matrix = competence_matrix
        self.tenure = tenure
        self.candidate_size = candidate_size
        self.nom = "RechercheTabou"

        self.current_order: Chromosome = create_random_solution(competence_matrix)
        self.best_order: Chromosome = [gene[:] for gene in self.current_order]
        self.makespan: int = evaluate_solution(self.best_order, competence_matrix)
        self.tabu_list: List[List[int]] = []
        self.iteration = 0
        self.historique_makespan: List[int] = [self.makespan]

    def step(self) -> None:
        if len(self.current_order) < 2:
            self.historique_makespan.append(self.makespan)
            self.iteration += 1
            return

        candidates = []
        for _ in range(self.candidate_size):
            i = random.randrange(len(self.current_order))
            j = random.randrange(len(self.current_order))
            if i == j:
                continue

            neighbor = [gene[:] for gene in self.current_order]
            moved = neighbor.pop(i)
            neighbor.insert(j, moved)
            ms = evaluate_solution(neighbor, self.competence_matrix)
            candidates.append((ms, neighbor, (i, j)))

        if not candidates:
            self.historique_makespan.append(self.makespan)
            self.iteration += 1
            return

        candidates.sort(key=lambda x: x[0])
        self.tabu_list = [entry for entry in self.tabu_list if entry[2] > self.iteration]

        for ms, neighbor, (i, j) in candidates:
            is_tabu = any(i == ti and j == tj for ti, tj, _ in self.tabu_list)
            if (not is_tabu) or (ms < self.makespan):
                self.current_order = neighbor
                self.tabu_list.append([i, j, self.iteration + self.tenure])
                if ms < self.makespan:
                    self.best_order = [gene[:] for gene in neighbor]
                    self.makespan = ms
                break

        self.iteration += 1
        self.historique_makespan.append(self.makespan)

    def reset(self, solution: Optional[Chromosome] = None) -> None:
        if solution:
            self.current_order = [gene[:] for gene in solution]
            self.best_order = [gene[:] for gene in solution]
            self.makespan = evaluate_solution(self.best_order, self.competence_matrix)
        self.tabu_list = []
        self.iteration = 0


class SousAgentGenetique:
    """Sous-agent genetique pour la diversification finale."""

    def __init__(
        self,
        competence_matrix: CompetenceMatrix,
        population_size: int = 20,
        mutation_rate: float = 0.2,
    ) -> None:
        self.competence_matrix = competence_matrix
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.nom = "AlgoGenetique"

        self.population: List[Chromosome] = [
            create_random_solution(competence_matrix) for _ in range(population_size)
        ]
        self.fitness_values, self.makespans = self._evaluate_population()

        best_idx = min(range(population_size), key=lambda i: self.makespans[i])
        self.best_order: Chromosome = [gene[:] for gene in self.population[best_idx]]
        self.makespan: int = self.makespans[best_idx]
        self.historique_makespan: List[int] = [self.makespan]

    def _evaluate_population(self) -> tuple[List[float], List[int]]:
        fitness_values: List[float] = []
        makespans: List[int] = []
        for chrom in self.population:
            ms = evaluate_solution(chrom, self.competence_matrix)
            makespans.append(ms)
            fitness_values.append(1.0 / (1.0 + ms))
        return fitness_values, makespans

    def _roulette_selection(self) -> Chromosome:
        total_fit = sum(self.fitness_values)
        if total_fit == 0:
            return [gene[:] for gene in random.choice(self.population)]

        target = random.random()
        cumulative = 0.0
        for chrom, fit in zip(self.population, self.fitness_values):
            cumulative += fit / total_fit
            if cumulative >= target:
                return [gene[:] for gene in chrom]
        return [gene[:] for gene in self.population[-1]]

    @staticmethod
    def _lox_crossover(parent_1: Chromosome, parent_2: Chromosome) -> Chromosome:
        size = len(parent_1)
        if size < 2:
            return [gene[:] for gene in parent_1]

        a, b = sorted(random.sample(range(size), 2))
        child: List[Optional[List[int]]] = [None] * size
        child[a : b + 1] = [gene[:] for gene in parent_1[a : b + 1]]

        p2_idx = 0
        for i in range(size):
            if child[i] is not None:
                continue
            while parent_2[p2_idx] in child:
                p2_idx += 1
            child[i] = parent_2[p2_idx][:]

        return [gene for gene in child if gene is not None]

    def _mutate(self, chrom: Chromosome) -> Chromosome:
        if len(chrom) < 2 or random.random() > self.mutation_rate:
            return [gene[:] for gene in chrom]

        child = [gene[:] for gene in chrom]
        i, j = random.sample(range(len(child)), 2)
        child[i], child[j] = child[j], child[i]
        return child

    def step(self) -> None:
        new_population: List[Chromosome] = [[gene[:] for gene in self.best_order]]
        while len(new_population) < self.population_size:
            parent_1 = self._roulette_selection()
            parent_2 = self._roulette_selection()
            child = self._lox_crossover(parent_1, parent_2)
            child = self._mutate(child)
            new_population.append(child)

        self.population = new_population
        self.fitness_values, self.makespans = self._evaluate_population()

        best_idx = min(range(len(self.population)), key=lambda i: self.makespans[i])
        if self.makespans[best_idx] < self.makespan:
            self.best_order = [gene[:] for gene in self.population[best_idx]]
            self.makespan = self.makespans[best_idx]

        self.historique_makespan.append(self.makespan)

    def inject_solution(self, solution: Chromosome, makespan: int) -> None:
        worst_idx = max(range(len(self.population)), key=lambda i: self.makespans[i])
        self.population[worst_idx] = [gene[:] for gene in solution]
        self.makespans[worst_idx] = makespan
        self.fitness_values[worst_idx] = 1.0 / (1.0 + makespan)
        if makespan < self.makespan:
            self.best_order = [gene[:] for gene in solution]
            self.makespan = makespan


class AgentAccueil:
    """Agent accueil des patients et priorisation par urgence."""

    def __init__(self, nom: str = "Accueil") -> None:
        self.nom = nom
        self.patients_en_attente: List[Patient] = []
        self.patients_en_cours: List[Patient] = []
        self.patients_termines: List[Patient] = []
        self.compteur_patients = 0
        self.boite_messages: List[Message] = []
        self.log: List[str] = []
        self.planning_actuel: Optional[Dict[str, Any]] = None

    @staticmethod
    def _analyser_competences(operations: List[List[int]]) -> Set[int]:
        competences: Set[int] = set()
        for operation in operations:
            for idx, val in enumerate(operation):
                if val > 0:
                    competences.add(idx)
        return competences

    def accueillir_patient(
        self,
        operations: List[List[int]],
        urgence: Optional[Urgence] = None,
        nom: Optional[str] = None,
        timestamp: float = 0.0,
    ) -> Patient:
        self.compteur_patients += 1

        if urgence is None:
            urgence = random.choices(
                list(Urgence),
                weights=[5, 15, 35, 30, 15],
                k=1,
            )[0]
        if nom is None:
            nom = f"Patient_{self.compteur_patients:03d}"

        competences_requises = self._analyser_competences(operations)

        patient = Patient(
            id=self.compteur_patients,
            nom=nom,
            urgence=urgence,
            heure_arrivee=timestamp,
            operations=operations,
            competences_requises=competences_requises,
        )

        self.patients_en_attente.append(patient)
        self.patients_en_attente.sort(key=lambda p: (p.urgence.value, p.heure_arrivee))

        self._log(
            f"[t={timestamp:.1f}] Patient {patient.nom} accueilli | "
            f"Urgence: {patient.urgence.name} | "
            f"{len(operations)} operations | "
            f"Competences requises: {sorted(competences_requises)}"
        )
        return patient

    def construire_matrice_competences(self) -> CompetenceMatrix:
        return [patient.operations for patient in self.patients_en_attente]

    def demander_ordonnancement(self, timestamp: float = 0.0) -> Message:
        matrice = self.construire_matrice_competences()
        toutes_competences: Set[int] = set()
        for patient in self.patients_en_attente:
            toutes_competences.update(patient.competences_requises)

        msg = Message(
            type=MessageType.LANCER_ORDONNANCEMENT,
            emetteur=self.nom,
            destinataire="Ordonnanceur",
            contenu={
                "competence_matrix": matrice,
                "nb_patients": len(self.patients_en_attente),
                "competences_requises": sorted(toutes_competences),
                "patients_info": [
                    {
                        "id": p.id,
                        "nom": p.nom,
                        "urgence": p.urgence.name,
                        "nb_ops": len(p.operations),
                    }
                    for p in self.patients_en_attente
                ],
            },
            timestamp=timestamp,
            priorite=min((p.urgence.value for p in self.patients_en_attente), default=5),
        )

        self._log(
            f"[t={timestamp:.1f}] Demande d'ordonnancement envoyee | "
            f"{len(self.patients_en_attente)} patients | "
            f"Competences necessaires: {sorted(toutes_competences)}"
        )
        return msg

    def signaler_nouveau_patient(self, patient: Patient, timestamp: float = 0.0) -> Message:
        return Message(
            type=MessageType.NOUVEAU_PATIENT,
            emetteur=self.nom,
            destinataire="Identificateur",
            contenu={
                "patient_id": patient.id,
                "patient_nom": patient.nom,
                "urgence": patient.urgence.name,
                "competences_requises": sorted(patient.competences_requises),
                "nb_operations": len(patient.operations),
            },
            timestamp=timestamp,
            priorite=patient.urgence.value,
        )

    def recevoir_planning(self, message: Message) -> None:
        self.planning_actuel = message.contenu
        self._log(
            f"[t={message.timestamp:.1f}] Planning recu | "
            f"CMax = {message.contenu.get('makespan', '?')}"
        )

        for patient in self.patients_en_attente:
            patient.statut = "en_cours"
            self.patients_en_cours.append(patient)
        self.patients_en_attente.clear()

    def get_etat(self) -> Dict[str, Any]:
        return {
            "agent": self.nom,
            "patients_en_attente": len(self.patients_en_attente),
            "patients_en_cours": len(self.patients_en_cours),
            "patients_termines": len(self.patients_termines),
            "total_accueillis": self.compteur_patients,
        }

    def afficher_file_attente(self) -> None:
        print("\n" + "=" * 60)
        print(f"  FILE D'ATTENTE - Agent {self.nom}")
        print("=" * 60)
        if not self.patients_en_attente:
            print("  (vide)")
        for idx, patient in enumerate(self.patients_en_attente, start=1):
            print(
                f"  {idx}. {patient.nom} | Urgence: {patient.urgence.name} | "
                f"{len(patient.operations)} ops | Arrivee: t={patient.heure_arrivee:.1f}"
            )
        print("=" * 60 + "\n")

    def _log(self, message: str) -> None:
        self.log.append(message)


class AgentIdentificateur:
    """Agent de gestion des ressources humaines (personnel/competences)."""

    def __init__(self, nom: str = "Identificateur") -> None:
        self.nom = nom
        self.registre_personnel: List[Personnel] = []
        self.boite_messages: List[Message] = []
        self.log: List[str] = []
        self.alertes: List[str] = []
        self.historique_absences: List[Dict[str, Any]] = []

    def enregistrer_personnel(self, nom: str, competences: List[int]) -> Personnel:
        personne = Personnel(
            id=len(self.registre_personnel) + 1,
            nom=nom,
            competences=sorted(competences),
        )
        self.registre_personnel.append(personne)
        self._log(f"Personnel enregistre: {nom} | Competences: {sorted(competences)}")
        return personne

    def generer_equipe(self, nb_personnel: int, nb_competences: int) -> None:
        roles = ["Dr.", "Inf.", "Aide", "Interne", "Spec.", "Chir.", "Anesth."]
        for idx in range(nb_personnel):
            nb_comp = random.randint(1, max(1, min(3, nb_competences)))
            comps = sorted(random.sample(range(nb_competences), nb_comp))
            role = random.choice(roles)
            suffix = f"{idx // 26 + 1}" if idx >= 26 else ""
            nom = f"{role} {chr(65 + (idx % 26))}{suffix}"
            self.enregistrer_personnel(nom, comps)
        self._log(f"Equipe de {nb_personnel} personnels generee")

    def verifier_disponibilite(self, competences_requises: Set[int]) -> Dict[str, Any]:
        disponibles = [
            p for p in self.registre_personnel if p.statut == StatutPersonnel.DISPONIBLE
        ]

        couverture: Dict[int, Dict[str, Any]] = {}
        for comp in sorted(competences_requises):
            personnels = [p for p in disponibles if comp in p.competences]
            couverture[comp] = {
                "couverte": len(personnels) > 0,
                "nb_disponibles": len(personnels),
                "personnel": [p.nom for p in personnels],
            }

        toutes_couvertes = all(info["couverte"] for info in couverture.values())

        rapport = {
            "toutes_couvertes": toutes_couvertes,
            "nb_personnel_disponible": len(disponibles),
            "nb_personnel_total": len(self.registre_personnel),
            "couverture_par_competence": couverture,
        }

        if not toutes_couvertes:
            manquantes = [comp for comp, info in couverture.items() if not info["couverte"]]
            alerte = f"Competences non couvertes: {manquantes}"
            self.alertes.append(alerte)
            self._log(alerte)

        return rapport

    def signaler_absence(self, personnel_id: int, timestamp: float = 0.0) -> Optional[Message]:
        personne = next((p for p in self.registre_personnel if p.id == personnel_id), None)
        if personne is None:
            self._log(f"Personnel ID={personnel_id} introuvable")
            return None

        personne.statut = StatutPersonnel.ABSENT
        self.historique_absences.append(
            {
                "personnel_id": personnel_id,
                "nom": personne.nom,
                "competences": list(personne.competences),
                "timestamp": timestamp,
            }
        )
        self._log(
            f"[t={timestamp:.1f}] ABSENCE: {personne.nom} | "
            f"Competences perdues: {personne.competences}"
        )

        competences_impactees: List[int] = []
        for comp in personne.competences:
            autres = [
                p
                for p in self.registre_personnel
                if p.statut == StatutPersonnel.DISPONIBLE and comp in p.competences
            ]
            if not autres:
                competences_impactees.append(comp)

        msg = Message(
            type=MessageType.RE_ORDONNANCEMENT,
            emetteur=self.nom,
            destinataire="Ordonnanceur",
            contenu={
                "raison": "absence_personnel",
                "personnel_absent": personne.nom,
                "competences_perdues": list(personne.competences),
                "competences_non_couvertes": competences_impactees,
                "impact_critique": len(competences_impactees) > 0,
            },
            timestamp=timestamp,
            priorite=1 if competences_impactees else 3,
        )
        self._log(
            "Re-ordonnancement declenche "
            f"(impact critique: {len(competences_impactees) > 0})"
        )
        return msg

    def repondre_demande_ressources(self, message: Message, timestamp: float = 0.0) -> Message:
        competences = set(message.contenu.get("competences_requises", []))
        rapport = self.verifier_disponibilite(competences)

        reponse = Message(
            type=MessageType.REPONSE_RESSOURCES,
            emetteur=self.nom,
            destinataire="Ordonnanceur",
            contenu=rapport,
            timestamp=timestamp,
        )
        self._log(
            f"[t={timestamp:.1f}] Reponse ressources envoyee | "
            f"{rapport['nb_personnel_disponible']}/{rapport['nb_personnel_total']} dispo"
        )
        return reponse

    def get_etat(self) -> Dict[str, Any]:
        status_counts = {
            statut.name: sum(1 for p in self.registre_personnel if p.statut == statut)
            for statut in StatutPersonnel
        }
        return {
            "agent": self.nom,
            "personnel_total": len(self.registre_personnel),
            "statuts": status_counts,
            "nb_alertes": len(self.alertes),
            "nb_absences": len(self.historique_absences),
        }

    def afficher_registre(self) -> None:
        print("\n" + "=" * 65)
        print(f"  REGISTRE DU PERSONNEL - Agent {self.nom}")
        print("=" * 65)
        for p in self.registre_personnel:
            comp = ", ".join(f"C{idx + 1}" for idx in p.competences)
            print(f"  {p.id:02d}. {p.nom:20s} | Comp: [{comp}] | {p.statut.value}")
        print("=" * 65 + "\n")

    def _log(self, message: str) -> None:
        self.log.append(message)


class AgentOrdonnanceur:
    """Agent ordonnanceur orchestrant les metaheuristiques."""

    def __init__(self, nom: str = "Ordonnanceur", steps_par_phase: int = 50) -> None:
        self.nom = nom
        self.competence_matrix: Optional[CompetenceMatrix] = None
        self.boite_messages: List[Message] = []
        self.log: List[str] = []
        self.steps_par_phase = steps_par_phase

        self.sa_agent: Optional[SousAgentRecuitSimule] = None
        self.tabu_agent: Optional[SousAgentTabou] = None
        self.ga_agent: Optional[SousAgentGenetique] = None

        self.meilleur_makespan: int = sys.maxsize
        self.meilleure_solution: Optional[Chromosome] = None
        self.historique_global: List[int] = []
        self.planning_decoded: Optional[List[List[Optional[List[int]]]]] = None

    def demander_ressources(self, competences_requises: List[int], timestamp: float = 0.0) -> Message:
        msg = Message(
            type=MessageType.DEMANDE_RESSOURCES,
            emetteur=self.nom,
            destinataire="Identificateur",
            contenu={"competences_requises": competences_requises},
            timestamp=timestamp,
        )
        self._log(
            f"[t={timestamp:.1f}] Demande de ressources envoyee | "
            f"Competences: {sorted(competences_requises)}"
        )
        return msg

    def ordonnancer(
        self,
        competence_matrix: CompetenceMatrix,
        mode: str = "pipeline",
        n_steps: int = 150,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        self.competence_matrix = competence_matrix
        self.historique_global = []

        if not competence_matrix:
            self.meilleur_makespan = 0
            self.meilleure_solution = []
            self.planning_decoded = []
            return {
                "makespan": self.meilleur_makespan,
                "solution": self.meilleure_solution,
                "planning": self.planning_decoded,
                "duree_calcul": 0.0,
                "historique": self.historique_global,
            }

        self.sa_agent = SousAgentRecuitSimule(competence_matrix)
        self.tabu_agent = SousAgentTabou(competence_matrix)
        self.ga_agent = SousAgentGenetique(competence_matrix)
        sous_agents = [self.sa_agent, self.tabu_agent, self.ga_agent]
        noms_phases = [
            "Recuit Simule (Exploration)",
            "Recherche Tabou (Intensification)",
            "Algo Genetique (Diversification)",
        ]

        self.meilleur_makespan = sys.maxsize
        self.meilleure_solution = None

        if verbose:
            print("\n" + "=" * 70)
            print(f"  ORDONNANCEMENT - Mode {mode.upper()}")
            print(f"  {len(competence_matrix)} patients | {n_steps} iterations")
            print("=" * 70)

        t_start = time.time()
        if mode == "pipeline":
            self._run_pipeline(sous_agents, noms_phases, verbose)
        else:
            self._run_parallele(sous_agents, n_steps, verbose)
        t_elapsed = time.time() - t_start

        if self.meilleure_solution is None:
            self.meilleure_solution = [gene[:] for gene in self.sa_agent.best_order]
            self.meilleur_makespan = self.sa_agent.makespan

        self.planning_decoded = decode_chromosome(self.meilleure_solution, competence_matrix)

        if verbose:
            print("\n" + "=" * 70)
            print(f"  RESULTAT : CMax = {self.meilleur_makespan} (en {t_elapsed:.2f}s)")
            print("=" * 70)

        self._log(
            f"Ordonnancement termine | CMax = {self.meilleur_makespan} | "
            f"Mode: {mode} | Duree: {t_elapsed:.2f}s"
        )

        return {
            "makespan": self.meilleur_makespan,
            "solution": self.meilleure_solution,
            "planning": self.planning_decoded,
            "duree_calcul": t_elapsed,
            "historique": self.historique_global,
        }

    def _run_pipeline(
        self,
        sous_agents: List[Any],
        noms_phases: List[str],
        verbose: bool,
    ) -> None:
        for phase_idx, (agent, nom_phase) in enumerate(zip(sous_agents, noms_phases)):
            if verbose:
                print(f"\n  Phase {phase_idx + 1}/3 : {nom_phase}")
                print("  " + "-" * 50)

            for step in range(self.steps_par_phase):
                agent.step()
                if agent.makespan < self.meilleur_makespan:
                    self.meilleur_makespan = agent.makespan
                    self.meilleure_solution = [gene[:] for gene in agent.best_order]
                self.historique_global.append(self.meilleur_makespan)

                if verbose and step % 10 == 0:
                    print(
                        f"    Step {step:3d} | Agent: {agent.makespan} | "
                        f"Global best: {self.meilleur_makespan}"
                    )

            if phase_idx < len(sous_agents) - 1 and self.meilleure_solution is not None:
                next_agent = sous_agents[phase_idx + 1]
                if hasattr(next_agent, "reset"):
                    next_agent.reset(self.meilleure_solution)
                elif hasattr(next_agent, "inject_solution"):
                    next_agent.inject_solution(self.meilleure_solution, self.meilleur_makespan)

                if verbose:
                    print(
                        f"    -> Transfert vers {noms_phases[phase_idx + 1]} "
                        f"(CMax={self.meilleur_makespan})"
                    )

    def _run_parallele(self, sous_agents: List[Any], n_steps: int, verbose: bool) -> None:
        for step in range(n_steps):
            for agent in sous_agents:
                agent.step()
                if agent.makespan < self.meilleur_makespan:
                    self.meilleur_makespan = agent.makespan
                    self.meilleure_solution = [gene[:] for gene in agent.best_order]

            self.historique_global.append(self.meilleur_makespan)

            if verbose and step % 20 == 0:
                agents_str = " | ".join(f"{a.nom}:{a.makespan}" for a in sous_agents)
                print(f"  Step {step:3d} | {agents_str} | Best: {self.meilleur_makespan}")

    def envoyer_planning(self, timestamp: float = 0.0) -> Message:
        return Message(
            type=MessageType.PLANNING_GENERE,
            emetteur=self.nom,
            destinataire="Accueil",
            contenu={
                "makespan": self.meilleur_makespan,
                "solution": self.meilleure_solution,
                "planning": self.planning_decoded,
            },
            timestamp=timestamp,
        )

    def get_etat(self) -> Dict[str, Any]:
        return {
            "agent": self.nom,
            "meilleur_makespan": self.meilleur_makespan,
            "nb_iterations": len(self.historique_global),
        }

    def _log(self, message: str) -> None:
        self.log.append(message)


class CoordinateurSMA:
    """Coordinateur du systeme multi-agents."""

    def __init__(
        self,
        nb_competences: int = 6,
        nb_personnel: int = 12,
        steps_par_phase: int = 50,
        seed: Optional[int] = None,
    ) -> None:
        if seed is not None:
            random.seed(seed)

        self.accueil = AgentAccueil()
        self.identificateur = AgentIdentificateur()
        self.ordonnanceur = AgentOrdonnanceur(steps_par_phase=steps_par_phase)

        self.nb_competences = nb_competences
        self.identificateur.generer_equipe(nb_personnel, nb_competences)

        self.messages_echanges: List[Message] = []
        self.log_global: List[str] = []
        self.timestamp = 0.0
        self.historique_plannings: List[Dict[str, Any]] = []

    def _envoyer_message(self, message: Message) -> None:
        self.messages_echanges.append(message)

        if message.destinataire == self.accueil.nom:
            self.accueil.boite_messages.append(message)
        elif message.destinataire == self.identificateur.nom:
            self.identificateur.boite_messages.append(message)
        elif message.destinataire == self.ordonnanceur.nom:
            self.ordonnanceur.boite_messages.append(message)

        self._log(str(message))

    def _log(self, message: str) -> None:
        self.log_global.append(message)

    def simuler_arrivee_patient(
        self,
        operations: List[List[int]],
        urgence: Optional[Urgence] = None,
        nom: Optional[str] = None,
    ) -> Patient:
        patient = self.accueil.accueillir_patient(
            operations=operations,
            urgence=urgence,
            nom=nom,
            timestamp=self.timestamp,
        )

        msg_nouveau = self.accueil.signaler_nouveau_patient(patient, self.timestamp)
        self._envoyer_message(msg_nouveau)

        rapport = self.identificateur.verifier_disponibilite(patient.competences_requises)
        if not rapport["toutes_couvertes"]:
            self._log(f"Ressources insuffisantes pour {patient.nom}")

        return patient

    def simuler_arrivees_batch(self, competence_matrix: CompetenceMatrix) -> None:
        for patient_ops in competence_matrix:
            self.timestamp += random.uniform(0.5, 3.0)
            self.simuler_arrivee_patient(patient_ops)
        self._log(f"{len(competence_matrix)} patients accueillis")

    def lancer_ordonnancement(
        self,
        mode: str = "pipeline",
        n_steps: int = 150,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        self._log("=" * 60)
        self._log(f"PROTOCOLE D'ORDONNANCEMENT LANCE (t={self.timestamp:.1f})")
        self._log("=" * 60)

        msg_ordo = self.accueil.demander_ordonnancement(self.timestamp)
        self._envoyer_message(msg_ordo)

        competences_requises = msg_ordo.contenu["competences_requises"]
        msg_ressources = self.ordonnanceur.demander_ressources(
            competences_requises=competences_requises,
            timestamp=self.timestamp,
        )
        self._envoyer_message(msg_ressources)

        msg_reponse = self.identificateur.repondre_demande_ressources(
            message=msg_ressources,
            timestamp=self.timestamp,
        )
        self._envoyer_message(msg_reponse)

        if verbose:
            rapport = msg_reponse.contenu
            print(
                f"\n  Ressources : {rapport['nb_personnel_disponible']}/"
                f"{rapport['nb_personnel_total']} personnels disponibles"
            )
            for comp, info in rapport["couverture_par_competence"].items():
                status = "OK" if info["couverte"] else "KO"
                personnes = ", ".join(info["personnel"]) if info["personnel"] else "-"
                print(
                    f"    {status} Competence C{comp + 1}: "
                    f"{info['nb_disponibles']} dispo ({personnes})"
                )

        matrice = msg_ordo.contenu["competence_matrix"]
        resultats = self.ordonnanceur.ordonnancer(
            competence_matrix=matrice,
            mode=mode,
            n_steps=n_steps,
            verbose=verbose,
        )

        msg_planning = self.ordonnanceur.envoyer_planning(self.timestamp)
        self._envoyer_message(msg_planning)
        self.accueil.recevoir_planning(msg_planning)

        self.historique_plannings.append(resultats)
        return resultats

    def simuler_absence(self, personnel_id: int, verbose: bool = True) -> Optional[Dict[str, Any]]:
        self.timestamp += 1.0
        if verbose:
            print("\n" + "!" * 60)
            print(f"  ABSENCE DETECTEE (t={self.timestamp:.1f})")
            print("!" * 60)

        msg_reordo = self.identificateur.signaler_absence(
            personnel_id=personnel_id,
            timestamp=self.timestamp,
        )
        if not msg_reordo:
            return None

        self._envoyer_message(msg_reordo)
        if verbose:
            contenu = msg_reordo.contenu
            print(f"  Personnel absent: {contenu['personnel_absent']}")
            print(f"  Competences perdues: {contenu['competences_perdues']}")
            print(f"  Impact critique: {contenu['impact_critique']}")

        if self.accueil.patients_en_attente:
            if verbose:
                print("  -> Relance de l'ordonnancement")
            return self.lancer_ordonnancement(mode="pipeline", n_steps=100, verbose=verbose)

        if verbose:
            print("  -> Pas de patients en attente, re-ordonnancement differe")
        return None

    def afficher_bilan(self) -> None:
        print("\n" + "=" * 70)
        print("  BILAN DU SYSTEME MULTI-AGENTS")
        print("=" * 70)

        for etat in [
            self.accueil.get_etat(),
            self.identificateur.get_etat(),
            self.ordonnanceur.get_etat(),
        ]:
            print(f"\n  {etat['agent']}:")
            for key, value in etat.items():
                if key != "agent":
                    print(f"    {key}: {value}")

        print(f"\n  Messages echanges: {len(self.messages_echanges)}")
        counts: Dict[str, int] = {}
        for msg in self.messages_echanges:
            counts[msg.type.value] = counts.get(msg.type.value, 0) + 1
        for msg_type, count in counts.items():
            print(f"    {msg_type}: {count}")
        print("=" * 70)


__all__ = [
    "Urgence",
    "MessageType",
    "StatutPersonnel",
    "Patient",
    "Personnel",
    "Message",
    "SousAgentRecuitSimule",
    "SousAgentTabou",
    "SousAgentGenetique",
    "AgentAccueil",
    "AgentIdentificateur",
    "AgentOrdonnanceur",
    "CoordinateurSMA",
]
