# Hetic_RL-FrozenLake

Implémentation d'un agent Q-Learning modulaire pour résoudre l'environnement FrozenLake de Gymnasium, incluant des scripts d'entraînement, d'évaluation et de visualisation des performances.

## Description

Ce projet implémente un agent d'apprentissage par renforcement utilisant l'algorithme Q-Learning pour naviguer dans l'environnement FrozenLake. L'objectif est d'apprendre à traverser un lac gelé en évitant les trous pour atteindre l'objectif.

### Objectif pédagogique

Ce projet illustre les concepts fondamentaux de l'apprentissage par renforcement :

- Stratégie epsilon-greedy (équilibre exploration/exploitation)
- Mise à jour Q-Learning
- Convergence d'un agent par l'expérience
- Reward shaping (modification des récompenses)
- Sauvegarde et chargement de modèles

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

#### Mode standard

Lancez le script principal pour entraîner l'agent :

```bash
python main.py
```

Le script va :

1. Créer un environnement FrozenLake 4x4 avec sol glissant
2. Charger l'agent existant (`SkaterAgent.pkl`) s'il existe, sinon en créer un nouveau
3. Entraîner l'agent sur 30 000 épisodes
4. Sauvegarder automatiquement le modèle à chaque amélioration du taux de succès
5. Afficher les graphiques de performance
6. Évaluer l'agent final sur 100 épisodes

#### Mode visualisation interactive

Pour observer l'entraînement en temps réel avec une interface graphique :

```bash
python main.py --visualize
```

ou

```bash
python main.py -v
```

Le mode visualisation affiche :

- **Grille FrozenLake** : Représentation visuelle de l'environnement avec l'agent en temps réel
- **Statistiques en direct** : Épisode actuel, epsilon, récompense, taux de succès
- **Graphique dynamique** : Évolution des récompenses sur les 100 derniers épisodes
- **Contrôles interactifs** : Gestion complète de l'entraînement en temps réel

**Contrôles disponibles :**

- `ESPACE` : Mettre en pause / Reprendre l'entraînement
- `R` : Réinitialiser l'entraînement (reset complet de l'agent)
- `T` : Basculer entre mode **Training** (exploration) et **Evaluation** (exploitation pure)
- `S` : Activer/Désactiver le sol glissant (`is_slippery`) - Force un reset de l'environnement
- `+` / `-` : Augmenter / Diminuer la vitesse d'entraînement
- `Q` ou `ESC` : Quitter le visualisateur

### Système de récompenses personnalisées

Le projet utilise un système de récompenses modifié pour accélérer l'apprentissage :

- **Trou (H)** : `-1` (pénalité)
- **Case vide (F)** : `0` (neutre)
- **Goal (G)** : `+1` (récompense)

Ce système permet à l'agent d'apprendre plus rapidement en recevant un signal négatif lorsqu'il tombe dans un trou, au lieu de simplement ne pas recevoir de récompense.

### Sauvegarde automatique

L'agent est automatiquement sauvegardé dans `SkaterAgent.pkl` :

- **Pendant l'entraînement** : Sauvegarde automatique chaque fois que le taux de succès s'améliore (après 100 épisodes)
- **À la fin** : Sauvegarde finale du modèle

### Chargement automatique

Au démarrage, le script :

1. Vérifie si `SkaterAgent.pkl` existe
2. Si oui, charge l'agent et **vérifie la compatibilité** (nombre d'états)
3. Si le modèle est incompatible avec l'environnement (ex: modèle 4x4 vs environnement 8x8), crée un nouvel agent
4. Permet de reprendre l'entraînement là où il s'est arrêté

Pour repartir de zéro, supprimez simplement le fichier :

```bash
# Windows
del SkaterAgent.pkl

# Linux/Mac
rm SkaterAgent.pkl
```

### Résultats attendus

#### Pendant l'entraînement

Le script affiche la progression tous les 1000 épisodes :

- Récompense moyenne sur les 100 derniers épisodes
- Valeur actuelle d'epsilon (taux d'exploration)
- Notifications de sauvegarde lors d'amélioration du taux de succès

#### Visualisations

Deux graphiques sont générés en mode standard :

1. **Récompenses par épisode** : Évolution brute des récompenses (peut être négative au début)
2. **Moyenne glissante** : Tendance lissée sur 100 épisodes

#### Évaluation finale

Le script affiche :

- **Taux de réussite** : Pourcentage d'épisodes où l'agent atteint l'objectif (80-85% attendu)
- **Récompense moyenne** : Performance moyenne sur 100 épisodes
- **Écart-type** : Variance des performances
- **Valeurs Q apprises** : Exemples de valeurs Q pour différents états

## Structure du projet

```
FrozenLakeML/
├── QLearningAgent.py          # Classe de l'agent Q-Learning
├── custom_reward_wrapper.py   # Wrapper pour récompenses personnalisées
├── visualizer.py              # Visualisateur interactif avec pygame
├── main.py                    # Script principal d'entraînement et évaluation
├── requirements.txt           # Dépendances Python
├── .gitignore                 # Fichiers à ignorer par Git
├── SkaterAgent.pkl            # Modèle sauvegardé (généré automatiquement)
└── README.md                  # Documentation
```

## Architecture

### QLearningAgent (`QLearningAgent.py`)

La classe `QLearningAgent` implémente :

- Initialisation de la Q-table (numpy array)
- Sélection d'action avec stratégie epsilon-greedy ou déterministe
- Mise à jour de la Q-table selon la formule Q-Learning
- Gestion correcte des états terminaux
- Décroissance du taux d'exploration
- Sauvegarde/chargement de l'agent avec pickle

### CustomRewardWrapper (`custom_reward_wrapper.py`)

Wrapper Gymnasium qui modifie les récompenses :

- Transforme les récompenses de l'environnement FrozenLake
- Applique le système de pénalités pour les trous
- Conserve la récompense positive pour le goal

### Visualisateur (`visualizer.py`)

La classe `TrainingVisualizer` offre :

- Affichage temps réel de la grille FrozenLake
- Panneau de statistiques dynamiques (épisode, epsilon, récompense, succès)
- Graphique des récompenses mis à jour en direct (matplotlib intégré dans pygame)
- Gestion des contrôles interactifs (pause, reset, vitesse, modes)
- Codes couleur : Bleu (safe), Noir (hole), Vert (goal), Orange (agent)
- Support des modes Training/Evaluation et Slippery on/off

### Script principal (`main.py`)

Le fichier `main.py` contient :

- `train_agent()` : Boucle d'entraînement avec support du visualisateur et sauvegarde auto
- `evaluate_agent()` : Évaluation sans exploration
- `plot_results()` : Visualisation des performances (mode standard)
- `main()` : Orchestration complète avec parsing d'arguments et gestion des modèles

## Paramètres d'entraînement

Les hyperparamètres optimisés sont :

- **Taux d'apprentissage (alpha)** : 0.1
- **Facteur de réduction (gamma)** : 0.99
- **Epsilon initial** : 1.0
- **Epsilon minimum** : 0.01
- **Décroissance epsilon** : 0.9995
- **Nombre d'épisodes** : 30 000

Ces paramètres ont été ajustés pour maximiser le taux de réussite sur l'environnement FrozenLake 4x4 avec sol glissant. Ils peuvent être modifiés dans la fonction `main()` de `main.py`.

### Explication des hyperparamètres

- **Alpha (0.1)** : Taux d'apprentissage modéré pour une convergence stable
- **Gamma (0.99)** : Facteur de discount élevé, essentiel pour les trajectoires longues (nécessité de planifier plusieurs pas à l'avance)
- **Epsilon decay (0.9995)** : Décroissance lente pour équilibrer exploration et exploitation
- **Epsilon min (0.01)** : Maintient 1% d'exploration pour éviter les blocages

### Performance attendue

Avec ces paramètres et le système de récompenses personnalisées :

- **Taux de succès** : 80-85% (limite théorique avec `is_slippery=True`)
- **Convergence** : Visible après ~5000-10000 épisodes
- **Stabilité** : Plateau atteint vers 15000-20000 épisodes

## Conseils d'utilisation

### Optimisation des performances

- Le mode visualisation ralentit l'entraînement. Pour un entraînement rapide, utilisez le mode standard.
- Ajustez la vitesse avec `+` / `-` en mode visualisation pour trouver le bon équilibre entre observation et performance.
- Les premiers 5000-10000 épisodes montrent l'apprentissage le plus visible. Utilisez le visualisateur sur cette période pour comprendre le comportement.

### Mode Evaluation vs Training

- **Mode Training** : L'agent explore (epsilon-greedy) et met à jour sa Q-table
- **Mode Evaluation** : L'agent exploite uniquement (greedy, deterministic), idéal pour observer la politique apprise
- Basculez avec `T` pour comparer les performances

### Sol glissant (Slippery)

- **is_slippery=True** : Actions stochastiques (33% chance de glisser), environnement difficile
- **is_slippery=False** : Actions déterministes, environnement facile (>95% de réussite possible)
- Basculez avec `S` pour tester l'agent dans différentes conditions

### Reproductibilité

- Les résultats peuvent varier légèrement entre les exécutions en raison de la nature stochastique de l'environnement.
- Pour des résultats reproductibles, définissez une graine aléatoire avec `np.random.seed()` et `env.reset(seed=X)`.

### Expérimentation

Pour tester différentes configurations :

1. **Modifier la taille de la grille** : Changez `map_name="4x4"` en `"8x8"` dans `main.py`
2. **Ajuster les récompenses** : Modifiez les valeurs dans `custom_reward_wrapper.py`
3. **Tester sans pénalités** : Commentez la ligne `env = CustomRewardWrapper(env)` dans `main.py`
4. **Changer les hyperparamètres** : Expérimentez avec alpha, gamma, epsilon_decay dans `main()`

## Limitations connues

- Le taux de succès maximal sur FrozenLake 4x4 avec `is_slippery=True` est limité à ~85% en raison de la stochasticité inhérente
- Le visualisateur peut ralentir significativement l'entraînement sur des machines moins puissantes
- La compatibilité des modèles sauvegardés est vérifiée uniquement sur le nombre d'états, pas sur d'autres paramètres

## Licence

Ce projet est développé à des fins pédagogiques dans le cadre du cours d'Optimisation par IA à HETIC.