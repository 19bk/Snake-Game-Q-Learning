import numpy as np
import random
import time
from collections import deque
import flet as ft

class SnakeGame:
    def __init__(self, width=15, height=10):
        self.width = width
        self.height = height
        self.reset()

    def reset(self):
        self.snake = [(self.height // 2, self.width // 2)]
        self.direction = (0, 1)
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

def draw_grid(page, game, cell_size):
    grid = []
    for row in range(game.height):
        grid_row = []
        for col in range(game.width):
            color = 'white'
            if (row, col) == game.food:
                color = 'red'
            elif (row, col) in game.snake:
                color = 'blue'
            grid_row.append(ft.Container(width=cell_size, height=cell_size, bgcolor=color, border_radius=5))
        grid.append(ft.Row(grid_row, spacing=2))
    return ft.Column(grid, spacing=2)

def draw_stats(agent, page):
    stats = agent.get_stats()
    stats_content = [
        ft.Text(f"Games Played: {stats['Games Played']}"),
        ft.Text(f"Current Score: {stats['Current Score']}"),
        ft.Text(f"Highest Score: {stats['Highest Score']}"),
        ft.Text(f"Avg Score (100): {stats['Avg Score (100)']:.2f}"),
        ft.Text(f"Total Food Eaten: {stats['Total Food Eaten']}"),
        ft.Text(f"Longest Snake: {stats['Longest Snake']}"),
        ft.Text(f"Learning Progress: {stats['Learning Progress']:.1f}%"),
        ft.ProgressBar(value=stats['Learning Progress'] / 100),
        ft.Text(f"Overall Accuracy: {stats['Overall Accuracy']:.1f}%"),
        ft.ProgressBar(value=stats['Overall Accuracy'] / 100),
        ft.Text(f"Recent Accuracy: {stats['Recent Accuracy']:.1f}%"),
        ft.ProgressBar(value=stats['Recent Accuracy'] / 100),
        ft.Text(f"Exploration Rate: {stats['Exploration Rate']:.1f}%"),
        ft.ProgressBar(value=stats['Exploration Rate'] / 100),
        ft.Text(f"Time: {stats['Time']}"),
    ]
    return ft.Column(stats_content, spacing=10)

def main(page):
    game = SnakeGame(width=15, height=10)
    agent = QLearningAgent(11, 3)
    cell_size = 30

    def update_ui():
        grid = draw_grid(page, game, cell_size)
        stats = draw_stats(agent, page)
        page.controls = [ft.Row([grid, stats], spacing=20)]
        page.update()

    def run_episode():
        state = game.reset()
        for _ in range(1000):
            action_idx = agent.get_action(state)
            action = np.eye(3)[action_idx]
            next_state, reward, done = game.step(action)
            agent.update(state, action_idx, reward, next_state, done)
            state = next_state
            update_ui()  # Update the UI to show the snake's movement
            time.sleep(0.1)  # Add delay to slow down the snake
            if done:
                break
        agent.update_stats(game.score)
        update_ui()

    def train():
        for episode in range(5000):
            run_episode()
            if episode % 100 == 0:
                stats = agent.get_stats()
                print(f"Episode: {episode}, Score: {game.score}, Progress: {stats['Learning Progress']:.2f}%")
            time.sleep(0.01)

    page.title = "Snake AI"
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.vertical_alignment = ft.MainAxisAlignment.CENTER
    update_ui()
    train()

ft.app(target=main)
