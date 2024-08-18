import numpy as np
import pygame
import random
import time
from collections import deque


class SnakeGame:
    def __init__(self, width=15, height=10):
        self.width = width
        self.height = height
        self.reset()

    def reset(self):
        self.snake = [(self.height // 2, self.width // 2)]
        self.direction = (0, 1)  # Start moving right
        self.food = self._place_food()
        self.score = 0
        self.steps = 0
        self.game_over = False
        return self._get_state()

    def _place_food(self):
        while True:
            food = (random.randint(0, self.height - 1), random.randint(0, self.width - 1))
            if food not in self.snake:
                return food

    def _get_state(self):
        head = self.snake[0]
        point_l = (head[0], head[1] - 1)
        point_r = (head[0], head[1] + 1)
        point_u = (head[0] - 1, head[1])
        point_d = (head[0] + 1, head[1])
        
        dir_l = self.direction == (0, -1)
        dir_r = self.direction == (0, 1)
        dir_u = self.direction == (-1, 0)
        dir_d = self.direction == (1, 0)

        state = [
            (dir_r and self._is_collision(point_r)) or 
            (dir_l and self._is_collision(point_l)) or 
            (dir_u and self._is_collision(point_u)) or 
            (dir_d and self._is_collision(point_d)),
            (dir_u and self._is_collision(point_r)) or 
            (dir_d and self._is_collision(point_l)) or 
            (dir_l and self._is_collision(point_u)) or 
            (dir_r and self._is_collision(point_d)),
            (dir_d and self._is_collision(point_r)) or 
            (dir_u and self._is_collision(point_l)) or 
            (dir_r and self._is_collision(point_u)) or 
            (dir_l and self._is_collision(point_d)),
            dir_l, dir_r, dir_u, dir_d,
            self.food[1] < head[1], self.food[1] > head[1],
            self.food[0] < head[0], self.food[0] > head[0]
        ]
        return np.array(state, dtype=int)

    def _is_collision(self, point):
        if point[0] < 0 or point[0] >= self.height or point[1] < 0 or point[1] >= self.width:
            return True
        if point in self.snake[1:]:
            return True
        return False

    def step(self, action):
        clock_wise = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        idx = clock_wise.index(self.direction)
        
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            new_dir = clock_wise[(idx + 1) % 4]
        else:
            new_dir = clock_wise[(idx - 1) % 4]

        self.direction = new_dir
        new_head = (self.snake[0][0] + self.direction[0], self.snake[0][1] + self.direction[1])

        self.steps += 1
        reward = 0
        self.game_over = self._is_collision(new_head)
        if self.game_over:
            reward = -10
            return self._get_state(), reward, self.game_over

        self.snake.insert(0, new_head)
        
        if new_head == self.food:
            self.score += 1
            reward = 10
            self.food = self._place_food()
        else:
            self.snake.pop()
        
        return self._get_state(), reward, self.game_over

class QLearningAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = 0.01
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.q_table = {}
        self.episode_count = 0
        self.highest_score = 0
        self.scores = deque(maxlen=100)
        self.action_history = deque(maxlen=1000)
        self.start_time = time.time()
        self.target_score = 50
        self.best_avg_score = 0
        self.good_moves = 0
        self.total_moves = 0
        self.move_history = deque(maxlen=1000)

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        state_tuple = tuple(state)
        return np.argmax(self.q_table.get(state_tuple, np.zeros(self.action_size)))

    def update(self, state, action, reward, next_state, done):
        state_tuple = tuple(state)
        next_state_tuple = tuple(next_state)
        if state_tuple not in self.q_table:
            self.q_table[state_tuple] = np.zeros(self.action_size)
        if next_state_tuple not in self.q_table:
            self.q_table[next_state_tuple] = np.zeros(self.action_size)
        
        q_value = self.q_table[state_tuple][action]
        max_next_q = np.max(self.q_table[next_state_tuple])
        new_q = q_value + self.learning_rate * (reward + self.gamma * max_next_q - q_value)
        self.q_table[state_tuple][action] = new_q

        self.action_history.append(action)
        
        self.total_moves += 1
        if reward > 0:
            self.good_moves += 1
        self.move_history.append(1 if reward > 0 else 0)

        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_stats(self):
        elapsed_time = time.time() - self.start_time
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)
        time_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
        
        avg_score = sum(self.scores) / len(self.scores) if self.scores else 0
        self.best_avg_score = max(self.best_avg_score, avg_score)
        
        learning_progress = (self.best_avg_score / self.target_score) * 100
        learning_progress = min(learning_progress, 100)
        
        accuracy = (self.good_moves / self.total_moves) * 100 if self.total_moves > 0 else 0
        recent_accuracy = (sum(self.move_history) / len(self.move_history)) * 100 if self.move_history else 0
        
        return {
            'Games Played': self.episode_count,
            'Current Score': self.scores[-1] if self.scores else 0,
            'Highest Score': self.highest_score,
            'Avg Score (100)': avg_score,
            'Total Food Eaten': sum(self.scores),
            'Longest Snake': max(self.scores) + 1 if self.scores else 1,
            'Learning Progress': learning_progress,
            'Exploration Rate': self.epsilon * 100,
            'Overall Accuracy': accuracy,
            'Recent Accuracy': recent_accuracy,
            'Time': time_str,
            'Action Distribution': np.bincount(self.action_history, minlength=self.action_size) / len(self.action_history) if self.action_history else np.zeros(self.action_size)
        }

    def update_stats(self, score):
        self.episode_count += 1
        self.scores.append(score)
        self.highest_score = max(self.highest_score, score)

class SnakeVisualization:
    def __init__(self, width, height, cell_size=30):
        self.width = width * cell_size
        self.height = height * cell_size
        self.cell_size = cell_size
        
        pygame.init()
        self.screen = pygame.display.set_mode((self.width + 600, self.height))
        pygame.display.set_caption("Snake Q-Learning Visualization")
        
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 20)
        
        self.colors = {
            'background': (40, 44, 52),
            'snake_body': (97, 175, 239),
            'snake_head': (65, 132, 215),
            'food': (224, 108, 117),
            'text': (171, 178, 191),
            'highlight': (255, 255, 0),
            'progress': (0, 191, 255),
            'sidebar': (33, 37, 43),
        }

    def draw(self, game, stats):
        self.screen.fill(self.colors['background'])
        self.draw_game_area(game)
        pygame.draw.rect(self.screen, self.colors['sidebar'], (self.width, 0, 600, self.height))
        self.draw_stats(stats)
        pygame.display.flip()

    def draw_game_area(self, game):
        # Draw food
        food_x = game.food[1] * self.cell_size + self.cell_size // 2
        food_y = game.food[0] * self.cell_size + self.cell_size // 2
        pygame.draw.circle(self.screen, self.colors['food'], 
                           (food_x, food_y), self.cell_size // 2 - 2)

        # Draw snake
        points = [(segment[1] * self.cell_size + self.cell_size // 2, 
                   segment[0] * self.cell_size + self.cell_size // 2) 
                  for segment in game.snake]
        
        # Draw the body
        if len(points) > 1:
            pygame.draw.lines(self.screen, self.colors['snake_body'], False, points, width=self.cell_size - 4)
        
        # Draw rounded ends for head and tail
        pygame.draw.circle(self.screen, self.colors['snake_head'], points[0], (self.cell_size - 4) // 2)
        pygame.draw.circle(self.screen, self.colors['snake_body'], points[-1], (self.cell_size - 4) // 2)
        
        # Add eyes to the head
        head_x, head_y = points[0]
        eye_radius = self.cell_size // 8
        eye_offset = self.cell_size // 4
        
        # Determine eye positions based on direction
        if len(game.snake) > 1:
            dx = game.snake[0][1] - game.snake[1][1]
            dy = game.snake[0][0] - game.snake[1][0]
            
            left_eye = (head_x - dy * eye_offset, head_y + dx * eye_offset)
            right_eye = (head_x + dy * eye_offset, head_y - dx * eye_offset)
            
            pygame.draw.circle(self.screen, (255, 255, 255), left_eye, eye_radius)
            pygame.draw.circle(self.screen, (255, 255, 255), right_eye, eye_radius)

    def draw_stats(self, stats):
        x1 = self.width + 20
        x2 = self.width + 320
        y = 20
        bar_width = 260
        bar_height = 20

        # Left column
        self.draw_text("Games Played:", stats['Games Played'], x1, y)
        y += 30
        self.draw_text("Current Score:", stats['Current Score'], x1, y)
        y += 30
        self.draw_text("Highest Score:", stats['Highest Score'], x1, y, self.colors['highlight'])
        y += 30
        self.draw_text("Avg Score (100):", f"{stats['Avg Score (100)']:.2f}", x1, y)
        y += 30
        self.draw_text("Total Food Eaten:", stats['Total Food Eaten'], x1, y)
        y += 30
        self.draw_text("Longest Snake:", stats['Longest Snake'], x1, y, self.colors['highlight'])
        y += 30
        self.draw_text("Time:", stats['Time'], x1, y)
        
        # Right column
        y = 20
        self.draw_progress_bar("Learning Progress", stats['Learning Progress'], x2, y, bar_width, bar_height)
        y += 50
        self.draw_progress_bar("Exploitation Rate", 100 - stats['Exploration Rate'], x2, y, bar_width, bar_height)
        y += 50
        self.draw_progress_bar("Exploration Rate", stats['Exploration Rate'], x2, y, bar_width, bar_height)
        y += 50
        self.draw_progress_bar("Overall Accuracy", stats['Overall Accuracy'], x2, y, bar_width, bar_height)
        y += 50
        self.draw_progress_bar("Recent Accuracy", stats['Recent Accuracy'], x2, y, bar_width, bar_height)
        
        # y += 50
        # self.draw_text("Action Distribution:", "", x2, y)
        # y += 30
        # actions = ['Straight', 'Right', 'Left']
        # for i, action in enumerate(actions):
        #     width = int(stats['Action Distribution'][i] * bar_width)
        #     pygame.draw.rect(self.screen, self.colors['progress'], (x2, y, width, bar_height))
        #     text = f"{action}: {stats['Action Distribution'][i]:.2f}"
        #     text_surf = self.small_font.render(text, True, self.colors['text'])
        #     text_rect = text_surf.get_rect(midleft=(x2 + 5, y + bar_height // 2))
        #     self.screen.blit(text_surf, text_rect)
        #     y += 30

    def draw_text(self, label, value, x, y, color=None):
        if color is None:
            color = self.colors['text']
        text = f"{label} {value}"
        self.screen.blit(self.font.render(text, True, color), (x, y))

    def draw_progress_bar(self, label, value, x, y, width, height):
        self.draw_text(f"{label}:", f"{value:.1f}%", x, y)
        y += 25
        pygame.draw.rect(self.screen, self.colors['text'], (x, y, width, height), 1)
        bar_width = int(value / 100 * width)
        pygame.draw.rect(self.screen, self.colors['progress'], (x, y, bar_width, height))

def train_and_visualize(episodes=5000, max_steps=1000, visualize_every=10):
    game = SnakeGame(width=15, height=10)
    agent = QLearningAgent(11, 3)
    vis = SnakeVisualization(game.width, game.height, cell_size=30)
    
    for episode in range(episodes):
        state = game.reset()
        total_reward = 0
        
        for step in range(max_steps):
            action_idx = agent.get_action(state)
            action = np.eye(3)[action_idx]
            next_state, reward, done = game.step(action)
            agent.update(state, action_idx, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            if episode % visualize_every == 0:
                stats = agent.get_stats()
                vis.draw(game, stats)
                pygame.time.wait(50)
            
            if done:
                break
        
        agent.update_stats(game.score)

        if episode % 100 == 0:
            stats = agent.get_stats()
            print(f"Episode: {episode}, Score: {game.score}, Progress: {stats['Learning Progress']:.2f}%")

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

    pygame.quit()

if __name__ == "__main__":
    train_and_visualize()