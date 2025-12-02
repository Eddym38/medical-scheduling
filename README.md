# Medical Scheduling Optimization

Projet d'optimisation d'ordonnancement multi-skills pour patients utilisant :
- Algorithme génétique
- Recuit simulé
- Recherche tabou

## Structure
- src/data : génération des patients
- src/algorithms : algorithmes d’optimisation
- src/decoding : transformation chromosome → planning
- src/evaluation : calcul du makespan
- src/visualization : affichage graphique

## Objectif
Minimiser le makespan sous contraintes :
- ordre patient
- ressources limitées
- multi-skills

## Execution
```bash
python src/main.py
