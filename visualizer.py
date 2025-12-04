import pygame
import numpy as np
from collections import deque
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO


class TrainingVisualizer:
    """Visualisateur interactif pour l'entraînement Q-Learning."""
    
    def __init__(self, env_desc, window_width=1200, window_height=700):
        """Initialise le visualisateur.
        
        Args:
            env_desc: Description de l'environnement FrozenLake
            window_width: Largeur de la fenêtre
            window_height: Hauteur de la fenêtre
        """
        pygame.init()
        self.window_width = window_width
        self.window_height = window_height
        self.screen = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption("FrozenLake Q-Learning - Training Visualizer")
        
        self.env_desc = env_desc
        self.grid_size = len(env_desc)
        
        self.cell_size = 100
        self.grid_offset_x = 50
        self.grid_offset_y = 50
        
        self.stats_x = 500
        self.stats_y = 50
        
        self.font_large = pygame.font.Font(None, 36)
        self.font_medium = pygame.font.Font(None, 28)
        self.font_small = pygame.font.Font(None, 20)
        
        self.colors = {
            'background': (240, 240, 245),
            'grid_line': (200, 200, 200),
            'frozen': (173, 216, 230),
            'hole': (50, 50, 50),
            'goal': (50, 205, 50),
            'start': (255, 215, 0),
            'text': (30, 30, 30),
            'text_secondary': (100, 100, 100)
        }
        
        self.paused = False
        self.speed = 1
        self.should_reset = False
        self.running = True
        
        self.rewards_history = deque(maxlen=100)
        self.episode = 0
        self.total_episodes = 0
        self.epsilon = 1.0
        self.current_reward = 0.0
        self.success_count = 0
        
        self.agent_pos = 0
        
        self.clock = pygame.time.Clock()

    def _draw_grid(self):
        """Dessine la grille FrozenLake."""
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x = self.grid_offset_x + j * self.cell_size
                y = self.grid_offset_y + i * self.cell_size
                
                cell = self.env_desc[i][j]
                
                if cell == b'S':
                    color = self.colors['start']
                elif cell == b'F':
                    color = self.colors['frozen']
                elif cell == b'H':
                    color = self.colors['hole']
                elif cell == b'G':
                    color = self.colors['goal']
                else:
                    color = self.colors['frozen']
                
                pygame.draw.rect(self.screen, color, (x, y, self.cell_size, self.cell_size))
                pygame.draw.rect(self.screen, self.colors['grid_line'], (x, y, self.cell_size, self.cell_size), 2)
                
                if cell == b'H':
                    text = self.font_small.render('HOLE', True, (255, 255, 255))
                    text_rect = text.get_rect(center=(x + self.cell_size//2, y + self.cell_size//2))
                    self.screen.blit(text, text_rect)
                elif cell == b'G':
                    text = self.font_small.render('GOAL', True, (255, 255, 255))
                    text_rect = text.get_rect(center=(x + self.cell_size//2, y + self.cell_size//2))
                    self.screen.blit(text, text_rect)

    def _draw_agent(self, state):
        """Dessine l'agent sur la grille.
        
        Args:
            state: Position actuelle de l'agent
        """
        row = state // self.grid_size
        col = state % self.grid_size
        
        x = self.grid_offset_x + col * self.cell_size + self.cell_size // 2
        y = self.grid_offset_y + row * self.cell_size + self.cell_size // 2
        
        pygame.draw.circle(self.screen, (255, 100, 0), (x, y), self.cell_size // 3)

    def _draw_stats(self):
        """Dessine le panneau de statistiques."""
        y_offset = self.stats_y
        line_height = 50
        
        title = self.font_large.render('STATISTIQUES', True, self.colors['text'])
        self.screen.blit(title, (self.stats_x, y_offset))
        y_offset += line_height + 10
        
        stats = [
            f"Épisode: {self.episode} / {self.total_episodes}",
            f"Epsilon: {self.epsilon:.3f}",
            f"Récompense: {self.current_reward:.2f}",
            f"Taux succès: {(self.success_count / max(1, self.episode) * 100):.1f}%",
            f"Vitesse: x{self.speed}"
        ]
        
        for stat in stats:
            text = self.font_medium.render(stat, True, self.colors['text'])
            self.screen.blit(text, (self.stats_x, y_offset))
            y_offset += line_height
        
        controls_x = self.stats_x + 350
        controls_y = self.stats_y + line_height + 10
        
        controls_title = self.font_medium.render('CONTRÔLES', True, self.colors['text'])
        self.screen.blit(controls_title, (controls_x, controls_y))
        controls_y += 40
        
        controls = [
            'ESPACE: Pause/Reprendre',
            'R: Reset',
            '+/-: Vitesse',
            'Q/ESC: Quitter'
        ]
        
        for control in controls:
            text = self.font_small.render(control, True, self.colors['text_secondary'])
            self.screen.blit(text, (controls_x, controls_y))
            controls_y += 30
        
        if self.paused:
            pause_text = self.font_large.render('⏸ PAUSE', True, (255, 0, 0))
            self.screen.blit(pause_text, (self.stats_x, y_offset + 30))

    def _draw_graph(self):
        """Dessine le graphique des récompenses."""
        if len(self.rewards_history) < 2:
            return
        
        graph_x = 50
        graph_y = 500
        graph_width = 1100
        graph_height = 150
        
        pygame.draw.rect(self.screen, (255, 255, 255), (graph_x, graph_y, graph_width, graph_height))
        pygame.draw.rect(self.screen, self.colors['grid_line'], (graph_x, graph_y, graph_width, graph_height), 2)
        
        title = self.font_medium.render('Récompenses (100 derniers épisodes)', True, self.colors['text'])
        self.screen.blit(title, (graph_x + 10, graph_y - 35))
        
        rewards_list = list(self.rewards_history)
        if len(rewards_list) > 1:
            max_reward = max(rewards_list) if max(rewards_list) > 0 else 1
            min_reward = min(rewards_list)
            
            points = []
            for i, reward in enumerate(rewards_list):
                x = graph_x + (i / len(rewards_list)) * graph_width
                normalized_reward = (reward - min_reward) / (max_reward - min_reward + 0.001)
                y = graph_y + graph_height - (normalized_reward * graph_height * 0.9) - graph_height * 0.05
                points.append((x, y))
            
            if len(points) > 1:
                pygame.draw.lines(self.screen, (0, 100, 255), False, points, 2)

    def handle_events(self):
        """Gère les événements pygame.
        
        Returns:
            True si le visualisateur doit continuer, False sinon
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return False
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    self.should_reset = True
                elif event.key == pygame.K_PLUS or event.key == pygame.K_KP_PLUS or event.key == pygame.K_EQUALS:
                    self.speed = min(50, self.speed + 10)
                elif event.key == pygame.K_MINUS or event.key == pygame.K_KP_MINUS:
                    self.speed = max(0.1, self.speed - 10)
                elif event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    self.running = False
                    return False
        
        return True

    def update(self, episode, total_episodes, epsilon, reward, state, success_count):
        """Met à jour l'affichage du visualisateur.
        
        Args:
            episode: Numéro de l'épisode actuel
            total_episodes: Nombre total d'épisodes
            epsilon: Valeur actuelle d'epsilon
            reward: Récompense de l'épisode
            state: État actuel de l'agent
            success_count: Nombre de succès total
        """
        self.episode = episode
        self.total_episodes = total_episodes
        self.epsilon = epsilon
        self.current_reward = reward
        self.agent_pos = state
        self.success_count = success_count
        
        self.rewards_history.append(reward)
        
        self.screen.fill(self.colors['background'])
        self._draw_grid()
        self._draw_agent(state)
        self._draw_stats()
        self._draw_graph()
        
        pygame.display.flip()
        self.clock.tick(60 * self.speed)

    def is_paused(self):
        """Vérifie si le visualisateur est en pause.
        
        Returns:
            True si en pause, False sinon
        """
        return self.paused

    def should_reset_training(self):
        """Vérifie si l'entraînement doit être réinitialisé.
        
        Returns:
            True si reset demandé, False sinon
        """
        if self.should_reset:
            self.should_reset = False
            self.rewards_history.clear()
            self.success_count = 0
            return True
        return False

    def is_running(self):
        """Vérifie si le visualisateur est actif.
        
        Returns:
            True si actif, False sinon
        """
        return self.running

    def close(self):
        """Ferme le visualisateur."""
        pygame.quit()

