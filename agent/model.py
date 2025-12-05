"""
Modèle Mesa pour la simulation multi-agent.
Coordonne les agents et gère l'environnement partagé.
"""
from mesa import Model
from mesa.datacollection import DataCollector
# Créer les agents
from genetic_agent import GeneticAgent
from simulated_annealing_agent import SimulatedAnnealingAgent
from tabu_agent import TabuAgent


class SchedulingModel(Model):
    """Modèle de simulation pour l'ordonnancement multi-agent."""

    def __init__(self, competence_matrix, n_genetic=1, n_simulated=1, n_tabu=1, 
                 mode="parallele", steps_par_phase=50):
        """
        Initialise le modèle.

        Args:
            competence_matrix: Matrice de compétences du problème
            n_genetic: Nombre d'agents génétiques
            n_simulated: Nombre d'agents recuit simulé
            n_tabu: Nombre d'agents Tabu
            mode: "parallele" (tous en même temps) ou "pipeline" (chaîne séquentielle)
            steps_par_phase: Nombre d'étapes par agent en mode pipeline
        """
        super().__init__()
        self.competence_matrix = competence_matrix
        self.mode = mode
        self.steps_par_phase = steps_par_phase
        self.current_step = 0

        # Meilleure solution globale trouvée par tous les agents
        self.global_best_solution = None
        self.global_best_makespan = float('inf')

        # Créer les agents dans l'ordre du pipeline: Simulated → Tabu → Genetic
        self.my_agents = []
        
        # Agent recuit simulé (exploration initiale)
        for i in range(n_simulated):
            agent = SimulatedAnnealingAgent(self, collaboratif=False)
            self.my_agents.append(agent)

        # Agent Tabu (intensification)
        for i in range(n_tabu):
            agent = TabuAgent(self, collaboratif=False)
            self.my_agents.append(agent)

        # Agent génétique (diversification finale)
        for i in range(n_genetic):
            agent = GeneticAgent(
                model=self, inner_population_size=20, collaboratif=False, mutation_rate=0.2)
            self.my_agents.append(agent)

        # Collecteur de données
        self.datacollector = DataCollector(
            model_reporters={
                "Best_Makespan": lambda m: m.global_best_makespan,
            }
        )

    def step(self):
        """Une étape de simulation."""
        if self.mode == "pipeline":
            self._step_pipeline()
        else:
            self._step_parallele()
        
        self.current_step += 1
        self.datacollector.collect(self)

    def _step_parallele(self):
        """Mode parallèle: tous les agents travaillent en même temps."""
        for agent in self.my_agents:
            agent.step()
            self.update_global_best(agent.best_order, agent.makespan)

    def _step_pipeline(self):
        """Mode pipeline: les agents travaillent en chaîne, passant leur solution au suivant."""
        # Déterminer quel agent est actif selon l'étape
        n_agents = len(self.my_agents)
        phase = self.current_step // self.steps_par_phase
        agent_index = min(phase, n_agents - 1)
        
        # Exécuter l'agent actif
        active_agent = self.my_agents[agent_index]
        active_agent.step()
        self.update_global_best(active_agent.best_order, active_agent.makespan)
        
        # Si on change de phase, passer la solution au prochain agent
        if (self.current_step + 1) % self.steps_par_phase == 0 and agent_index < n_agents - 1:
            next_agent = self.my_agents[agent_index + 1]
            self._transfer_solution(active_agent, next_agent)
            print(f"\n>>> Transfert: {type(active_agent).__name__} (makespan={active_agent.makespan}) → {type(next_agent).__name__}")

    def _transfer_solution(self, from_agent, to_agent):
        """Transfère la meilleure solution d'un agent vers un autre."""
        solution_copy = [gene[:] for gene in from_agent.best_order]
        
        # Adapter selon le type d'agent destinataire
        if isinstance(to_agent, TabuAgent):
            to_agent.current_order = solution_copy
            to_agent.best_order = [gene[:] for gene in solution_copy]
            to_agent.makespan = from_agent.makespan
            to_agent.tabu_list = []  # Reset la liste tabu
            
        elif isinstance(to_agent, GeneticAgent):
            # Injecter la solution dans la population génétique
            to_agent.best_order = solution_copy
            to_agent.makespan = from_agent.makespan
            # Remplacer le pire individu par cette solution
            worst_idx = max(range(len(to_agent.inner_population)), 
                           key=lambda i: to_agent.makespans[i])
            to_agent.inner_population[worst_idx] = solution_copy
            to_agent.makespans[worst_idx] = from_agent.makespan
            to_agent.fitness_values[worst_idx] = 1 / (1 + from_agent.makespan)
            
        elif isinstance(to_agent, SimulatedAnnealingAgent):
            to_agent.best_order = solution_copy
            to_agent.makespan = from_agent.makespan
            # Réinitialiser la température pour une nouvelle exploration
            to_agent.temperature = to_agent.temp_init

    def update_global_best(self, solution, makespan):
        """
        Met à jour la meilleure solution globale si nécessaire.

        Args:
            solution: Liste d'opérations
            makespan: Makespan de la solution

        Returns:
            True si c'est une amélioration
        """
        if makespan < self.global_best_makespan:
            self.global_best_solution = solution
            self.global_best_makespan = makespan
            return True
        return False
