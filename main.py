import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from QLearningAgent import QLearningAgent


def train_agent(env, agent, n_episodes, alpha, gamma, epsilon_decay, epsilon_min):
    """Entraîne l'agent sur l'environnement.
    
    Args:
        env: Environnement Gymnasium
        agent: Agent Q-Learning
        n_episodes: Nombre d'épisodes d'entraînement
        alpha: Taux d'apprentissage
        gamma: Facteur de réduction
        epsilon_decay: Facteur de décroissance d'epsilon
        epsilon_min: Valeur minimale d'epsilon
        
    Returns:
        Liste des récompenses par épisode
    """
    rewards_history = []
    success_count = 0

    for episode in range(n_episodes):
        state = env.reset()[0]
        done = False
        total_reward = 0

        while not done:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            agent.update(state, action, reward, next_state, done, alpha, gamma)
            
            state = next_state
            total_reward += reward

        rewards_history.append(total_reward)
        if total_reward > 0:
            success_count += 1

        agent.decay_epsilon(epsilon_decay, epsilon_min)

        if (episode + 1) % 1000 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            print(f"Épisode {episode + 1}/{n_episodes} - Récompense moyenne (100 derniers): {avg_reward:.2f} - ε: {agent.epsilon:.3f}")

    return rewards_history


def plot_results(rewards_history, window_size=100):
    """Affiche les courbes de récompenses.
    
    Args:
        rewards_history: Liste des récompenses par épisode
        window_size: Taille de la fenêtre pour la moyenne glissante
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(rewards_history, alpha=0.3)
    ax1.set_xlabel('Épisode')
    ax1.set_ylabel('Récompense')
    ax1.set_title('Récompenses par épisode')
    ax1.grid(True)

    moving_avg = np.convolve(rewards_history, np.ones(window_size)/window_size, mode='valid')
    ax2.plot(moving_avg)
    ax2.set_xlabel('Épisode')
    ax2.set_ylabel('Récompense moyenne')
    ax2.set_title(f'Moyenne glissante (fenêtre={window_size})')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def evaluate_agent(env, agent, n_episodes=100):
    """Évalue l'agent sans exploration.
    
    Args:
        env: Environnement Gymnasium
        agent: Agent Q-Learning
        n_episodes: Nombre d'épisodes d'évaluation
        
    Returns:
        Tuple (taux de réussite, récompense moyenne, écart-type)
    """
    rewards = []
    success_count = 0

    for _ in range(n_episodes):
        state = env.reset()[0]
        done = False
        total_reward = 0

        while not done:
            action = agent.get_action(state, deterministic=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            state = next_state
            total_reward += reward

        rewards.append(total_reward)
        if total_reward > 0:
            success_count += 1

    success_rate = success_count / n_episodes
    avg_reward = np.mean(rewards)
    std_reward = np.std(rewards)

    return success_rate, avg_reward, std_reward


def main():
    """Fonction principale du programme."""
    env = gym.make(
        'FrozenLake-v1',
        desc=None,
        map_name="4x4",
        is_slippery=True,
        success_rate=1.0/3.0,
        reward_schedule=(1, 0, 0)
    )

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    print(f"Taille de l'espace d'observation : {n_states}")
    print(f"Nombre d'actions : {n_actions}")

    agent = QLearningAgent(n_states, n_actions)

    alpha = 0.1
    gamma = 0.99
    epsilon_decay = 0.995
    epsilon_min = 0.01
    n_episodes = 20000

    print("\n" + "="*50)
    print("DÉBUT DE L'ENTRAÎNEMENT")
    print("="*50)

    rewards_history = train_agent(
        env, agent, n_episodes, alpha, gamma, epsilon_decay, epsilon_min
    )

    agent.save('SkaterAgent.pkl')

    print("\n" + "="*50)
    print("VISUALISATION DES RÉSULTATS")
    print("="*50)

    plot_results(rewards_history)

    print("\n" + "="*50)
    print("ÉVALUATION FINALE")
    print("="*50)

    success_rate, avg_reward, std_reward = evaluate_agent(env, agent)

    print(f"\nTaux de réussite : {success_rate*100:.1f}%")
    print(f"Récompense moyenne : {avg_reward:.3f}")
    print(f"Écart-type : {std_reward:.3f}")

    print("\nExemples de valeurs Q apprises :")
    print(f"État 0 (Start): {agent.Q[0]}")
    print(f"État 15 (Goal): {agent.Q[15]}")
    print(f"État 5 (Trou): {agent.Q[5]}")
    print(f"État 6 (Passage): {agent.Q[6]}")

    env.close()


if __name__ == "__main__":
    main()

