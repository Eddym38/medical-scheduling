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
```

## Interface Streamlit (SMA metier)

```bash
streamlit run app_streamlit.py
```

Fonctionnalites:
- generation d'un scenario complet (nb patients, nb personnel, cadence d'arrivee)
- simulation d'absences pendant la journee
- simulation live (timeline, pas a pas, autoplay, execution complete)
- bouton unique "Generer scenario" qui calcule un resume final puis permet le replay live
- mode live avec lecture selon ecarts temporels d'arrivee (vitesse reglable)
- ajout de patients (manuel ou aleatoire)
- edition personnel (statut, competences)
- ordonnancement pipeline/parallele
- absence instantanee et re-ordonnancement
- visualisation du planning final
