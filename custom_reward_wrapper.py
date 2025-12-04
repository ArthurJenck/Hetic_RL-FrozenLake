import gymnasium as gym


class CustomRewardWrapper(gym.Wrapper):
    """Wrapper pour personnaliser les récompenses de FrozenLake.
    
    Récompenses modifiées :
    - Trou : -1
    - Case vide : 0
    - Goal : +1
    """
    
    def __init__(self, env):
        """Initialise le wrapper.
        
        Args:
            env: L'environnement Gymnasium à wrapper.
        """
        super().__init__(env)
    
    def step(self, action):
        """Exécute une action et modifie la récompense.
        
        Args:
            action: L'action à exécuter.
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        if terminated:
            if reward == 0:
                reward = -1
            elif reward == 1:
                reward = 1
        else:
            reward = 0
        
        return observation, reward, terminated, truncated, info

