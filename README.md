# Hetic_RL-FrozenLake

Implémentation d'un agent Q-Learning modulaire pour résoudre l'environnement FrozenLake de Gymnasium, incluant des scripts d'entraînement, d'évaluation et de visualisation des performances.

## Description

Ce projet implémente un agent d'apprentissage par renforcement utilisant l'algorithme Q-Learning pour naviguer dans l'environnement FrozenLake. L'objectif est d'apprendre à traverser un lac gelé en évitant les trous pour atteindre l'objectif.

### Objectif pédagogique

Ce projet illustre les concepts fondamentaux de l'apprentissage par renforcement :

- Stratégie epsilon-greedy (équilibre exploration/exploitation)
- Mise à jour Q-Learning
- Convergence d'un agent par l'expérience

## Installation

### Prérequis

- Python 3.8+
- pip

### Dépendances

Installez les dépendances nécessaires :

```bash
pip install -r requirements.txt
```

Les packages suivants seront installés :

- `gymnasium` : Environnement de simulation
- `pygame` : Moteur de rendu graphique
- `numpy` : Calculs numériques
- `matplotlib` : Visualisation des résultats

## Utilisation

### Entraînement de l'agent

Lancez le script principal pour entraîner l'agent :

```bash
python main.py
```

Le script va :

1. Créer un environnement FrozenLake 4x4 avec sol glissant
2. Entraîner l'agent sur 20 000 épisodes
3. Sauvegarder l'agent entraîné dans `SkaterAgent.pkl`
4. Afficher les graphiques de performance
5. Évaluer l'agent final sur 100 épisodes

### Résultats attendus

#### Pendant l'entraînement

Le script affiche la progression tous les 1000 épisodes :

- Récompense moyenne sur les 100 derniers épisodes
- Valeur actuelle d'epsilon (taux d'exploration)

#### Visualisations

Deux graphiques sont générés :

1. **Récompenses par épisode** : Évolution brute des récompenses
2. **Moyenne glissante** : Tendance lissée sur 100 épisodes

#### Évaluation finale

Le script affiche :

- **Taux de réussite** : Pourcentage d'épisodes où l'agent atteint l'objectif
- **Récompense moyenne** : Performance moyenne sur 100 épisodes
- **Écart-type** : Variance des performances
- **Valeurs Q apprises** : Exemples de valeurs Q pour différents états

Un agent correctement entraîné devrait atteindre un taux de réussite entre 70% et 80% sur cet environnement stochastique.

## Structure du projet

```md
FrozenLakeML/
├── QLearningAgent.py    # Classe de l'agent Q-Learning
├── main.py              # Script principal d'entraînement et évaluation
├── requirements.txt     # Dépendances Python
├── .gitignore          # Fichiers à ignorer par Git
└── README.md           # Documentation
```

## Architecture

### QLearningAgent

La classe `QLearningAgent` implémente :

- Initialisation de la Q-table
- Sélection d'action avec stratégie epsilon-greedy
- Mise à jour de la Q-table selon la formule Q-Learning
- Décroissance du taux d'exploration
- Sauvegarde/chargement de l'agent

### Script principal (main.py)

Le fichier `main.py` contient :

- `train_agent()` : Boucle d'entraînement
- `evaluate_agent()` : Évaluation sans exploration
- `plot_results()` : Visualisation des performances
- `main()` : Orchestration complète

## Paramètres d'entraînement

Les hyperparamètres par défaut sont :

- **Taux d'apprentissage (alpha)** : 0.1
- **Facteur de réduction (gamma)** : 0.99
- **Epsilon initial** : 1.0
- **Epsilon minimum** : 0.01
- **Décroissance epsilon** : 0.995
- **Nombre d'épisodes** : 20 000

Ces paramètres peuvent être ajustés dans la fonction `main()`.
