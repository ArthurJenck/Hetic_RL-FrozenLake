import numpy as np
import pickle


class QLearningAgent:
    """Agent d'apprentissage par renforcement utilisant Q-Learning."""

    def __init__(self, n_states, n_actions):
        """Initialise l'agent Q-Learning.
        
        Args:
            n_states: Nombre d'états possibles dans l'environnement
            n_actions: Nombre d'actions possibles
        """
        self.Q = np.zeros((n_states, n_actions))
        self.epsilon = 1.0
        self.n_actions = n_actions

    def get_action(self, state, deterministic=False):
        """Sélectionne une action selon la stratégie epsilon-greedy.
        
        Args:
            state: État actuel
            deterministic: Si True, choisit toujours la meilleure action (pas d'exploration)
            
        Returns:
            Index de l'action sélectionnée
        """
        if deterministic or np.random.random() > self.epsilon:
            return np.argmax(self.Q[state, :])
        else:
            return np.random.randint(0, self.n_actions)

    def update(self, state, action, reward, next_state, done, alpha=0.1, gamma=0.99):
        """Met à jour la Q-table selon la formule de Q-Learning.
        
        Args:
            state: État actuel
            action: Action effectuée
            reward: Récompense reçue
            next_state: État suivant atteint
            done: Indique si l'épisode est terminé
            alpha: Taux d'apprentissage
            gamma: Facteur de réduction
        """
        best_next_action = np.max(self.Q[next_state, :])
        td_target = reward + gamma * best_next_action
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += alpha * td_error

    def decay_epsilon(self, epsilon_decay=0.995, epsilon_min=0.01):
        """Réduit le taux d'exploration.
        
        Args:
            epsilon_decay: Facteur de décroissance multiplicatif
            epsilon_min: Valeur minimale d'epsilon
        """
        self.epsilon = max(epsilon_min, self.epsilon * epsilon_decay)

    def save(self, filepath):
        """Sauvegarde l'agent dans un fichier.
        
        Args:
            filepath: Chemin où sauvegarder l'agent
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Agent sauvegardé dans {filepath}")

    @classmethod
    def load(cls, filepath):
        """Charge un agent depuis un fichier.
        
        Args:
            filepath: Chemin vers le fichier de l'agent sauvegardé
            
        Returns:
            Instance de QLearningAgent chargée
        """
        with open(filepath, 'rb') as f:
            agent = pickle.load(f)
        print(f"Agent chargé depuis {filepath}")
        return agent

