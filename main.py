import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import argparse
import os
from QLearningAgent import QLearningAgent
from visualizer import TrainingVisualizer
from custom_reward_wrapper import CustomRewardWrapper


def train_agent(env, agent, n_episodes, alpha, gamma, epsilon_decay, epsilon_min, visualizer=None, is_slippery=True):
    """Entraîne l'agent sur l'environnement.
    
    Args:
        env: Environnement Gymnasium
        agent: Agent Q-Learning
        n_episodes: Nombre d'épisodes d'entraînement
        alpha: Taux d'apprentissage
        gamma: Facteur de réduction
        epsilon_decay: Facteur de décroissance d'epsilon
        epsilon_min: Valeur minimale d'epsilon
        visualizer: Visualisateur optionnel
        is_slippery: État initial du slippery
        
    Returns:
        Tuple (liste des récompenses, environnement final, agent final)
    """
    rewards_history = []
    success_count = 0
    current_is_slippery = is_slippery
    best_success_rate = 0.0
    eval_window = 100

    for episode in range(n_episodes):
        if visualizer:
            if not visualizer.handle_events() or not visualizer.is_running():
                break
            
            if visualizer.is_slippery_toggle_requested():
                current_is_slippery = not current_is_slippery
                env.close()
                env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=current_is_slippery)
                env = CustomRewardWrapper(env)
                agent = QLearningAgent(env.observation_space.n, env.action_space.n)
                visualizer.set_slippery(current_is_slippery)
                rewards_history = []
                success_count = 0
                episode = 0
                continue
            
            if visualizer.should_reset_training():
                agent = QLearningAgent(env.observation_space.n, env.action_space.n)
                rewards_history = []
                success_count = 0
                episode = 0
                continue
            
            while visualizer.is_paused():
                visualizer.handle_events()
                if not visualizer.is_running():
                    return rewards_history
        
        state = env.reset()[0]
        done = False
        total_reward = 0

        while not done:
            deterministic = visualizer.is_evaluation_mode() if visualizer else False
            action = agent.get_action(state, deterministic=deterministic)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            if not (visualizer and visualizer.is_evaluation_mode()):
                agent.update(state, action, reward, next_state, done, alpha, gamma)
            
            state = next_state
            total_reward += reward

        rewards_history.append(total_reward)
        if total_reward > 0:
            success_count += 1
        
        if episode >= eval_window:
            current_success_rate = success_count / (episode + 1)
            if current_success_rate > best_success_rate:
                best_success_rate = current_success_rate
                agent.save('SkaterAgent.pkl')
                if not visualizer:
                    print(f"[Épisode {episode + 1}] Nouveau meilleur taux : {best_success_rate*100:.1f}% - Modèle sauvegardé")
        
        if visualizer:
            visualizer.update(episode + 1, n_episodes, agent.epsilon, total_reward, state, success_count)

        if not (visualizer and visualizer.is_evaluation_mode()):
            agent.decay_epsilon(epsilon_decay, epsilon_min)

        if (episode + 1) % 1000 == 0 and not visualizer:
            avg_reward = np.mean(rewards_history[-100:])
            print(f"Épisode {episode + 1}/{n_episodes} - Récompense moyenne (100 derniers): {avg_reward:.2f} - ε: {agent.epsilon:.3f}")

    return rewards_history, env, agent


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
    parser = argparse.ArgumentParser(description='Entraînement Q-Learning sur FrozenLake')
    parser.add_argument('-v', '--visualize', action='store_true', 
                        help='Activer le mode visualisation interactive')
    args = parser.parse_args()
    
    env = gym.make(
        'FrozenLake-v1',
        map_name="4x4",
        is_slippery=True
    )
    env = CustomRewardWrapper(env)

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    if not args.visualize:
        print(f"Taille de l'espace d'observation : {n_states}")
        print(f"Nombre d'actions : {n_actions}")

    model_path = 'SkaterAgent.pkl'
    if os.path.exists(model_path):
        print(f"\nModèle existant détecté : {model_path}")
        temp_agent = QLearningAgent.load(model_path)
        
        if temp_agent.Q.shape[0] != n_states:
            print(f"⚠️  INCOMPATIBILITÉ : Le modèle a {temp_agent.Q.shape[0]} états, l'environnement en a {n_states}")
            print("→ Création d'un nouvel agent au lieu du chargement\n")
            agent = QLearningAgent(n_states, n_actions)
        else:
            agent = temp_agent
            print("Reprise de l'entraînement depuis le modèle sauvegardé\n")
    else:
        print("\nAucun modèle existant - Initialisation d'un nouvel agent\n")
        agent = QLearningAgent(n_states, n_actions)

    alpha = 0.1
    gamma = 0.99
    epsilon_decay = 0.9995
    epsilon_min = 0.01
    n_episodes = 30000

    visualizer = None
    if args.visualize:
        env_unwrapped = env.unwrapped
        visualizer = TrainingVisualizer(env_unwrapped.desc)
        print("Mode visualisation activé")
        print("Contrôles: ESPACE=Pause, R=Reset, +/-=Vitesse, T=Training/Eval, S=Slippery, Q/ESC=Quitter")
    else:
        print("\n" + "="*50)
        print("DÉBUT DE L'ENTRAÎNEMENT")
        print("="*50)

    rewards_history, env, agent = train_agent(
        env, agent, n_episodes, alpha, gamma, epsilon_decay, epsilon_min, visualizer, is_slippery=True
    )

    if visualizer:
        visualizer.close()

    print(f"\nSauvegarde finale du modèle...")
    agent.save('SkaterAgent.pkl')

    if not args.visualize:
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

