import sys
import numpy as np
import random
from collections import deque
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGraphicsView, QGraphicsScene, QGraphicsRectItem
from PyQt5.QtGui import QPainter, QBrush, QColor, QPen
from PyQt5.QtCore import Qt, QTimer, QRectF

class SnakeGame:
    def __init__(self, width=20, height=15):
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
        point_l = (head[0], (head[1] - 1) % self.width)
        point_r = (head[0], (head[1] + 1) % self.width)
        point_u = ((head[0] - 1) % self.height, head[1])
        point_d = ((head[0] + 1) % self.height, head[1])
        
        dir_l = self.direction == (0, -1)
        dir_r = self.direction == (0, 1)
        dir_u = self.direction == (-1, 0)
        dir_d = self.direction == (1, 0)

        state = [
            (dir_r and point_r in self.snake) or (dir_l and point_l in self.snake) or 
            (dir_u and point_u in self.snake) or (dir_d and point_d in self.snake),
            (dir_u and point_r in self.snake) or (dir_d and point_l in self.snake) or 
            (dir_l and point_u in self.snake) or (dir_r and point_d in self.snake),
            (dir_d and point_r in self.snake) or (dir_u and point_l in self.snake) or 
            (dir_r and point_u in self.snake) or (dir_l and point_d in self.snake),
            dir_l, dir_r, dir_u, dir_d,
            self.food[1] < head[1], self.food[1] > head[1],
            self.food[0] < head[0], self.food[0] > head[0]
        ]
        return np.array(state, dtype=int)

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
        new_head = ((self.snake[0][0] + self.direction[0]) % self.height, 
                    (self.snake[0][1] + self.direction[1]) % self.width)

        self.steps += 1
        reward = 0
        self.game_over = new_head in self.snake[1:]
        if self.game_over:
            reward = -10
        else:
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
        self.total_reward = 0
        self.highest_score = 0
        self.food_eaten = 0
        self.total_steps = 0
        self.longest_snake = 1
        self.action_history = deque(maxlen=1000)
        self.scores = deque(maxlen=100)
        self.current_q_value = 0

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
        self.current_q_value = new_q

        self.action_history.append(action)
        self.total_reward += reward

        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_stats(self):
        return {
            'Episode': self.episode_count,
            'Current Score': self.scores[-1] if self.scores else 0,
            'Highest Score': self.highest_score,
            'Avg Score (100)': sum(self.scores) / len(self.scores) if self.scores else 0,
            'Epsilon': self.epsilon,
            'Current Q-Value': self.current_q_value,
            'Learning Rate': self.learning_rate,
            'Total Reward': self.total_reward,
            'Food Eaten': self.food_eaten,
            'Avg Steps per Food': self.total_steps / self.food_eaten if self.food_eaten else 0,
            'Longest Snake': self.longest_snake,
            'Action Distribution': np.bincount(self.action_history, minlength=self.action_size) / len(self.action_history) if self.action_history else np.zeros(self.action_size)
        }

class SnakeVisualization(QMainWindow):
    def __init__(self, game, agent):
        super().__init__()
        self.game = game
        self.agent = agent
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Snake Q-Learning Visualization')
        self.setGeometry(100, 100, 800, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Game board
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setFixedSize(400, 300)
        main_layout.addWidget(self.view)

        # Stats panel
        stats_widget = QWidget()
        stats_layout = QVBoxLayout(stats_widget)
        main_layout.addWidget(stats_widget)

        self.stats_labels = {}
        for stat in self.agent.get_stats().keys():
            label = QLabel(f"{stat}: 0")
            stats_layout.addWidget(label)
            self.stats_labels[stat] = label

        # Timer for updates
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_visualization)
        self.timer.start(100)  # Update every 100ms

    def update_visualization(self):
        self.update_game_board()
        self.update_stats()

    def update_game_board(self):
        self.scene.clear()
        cell_size = 15

        # Draw snake
        for segment in self.game.snake:
            self.scene.addRect(QRectF(segment[1] * cell_size, segment[0] * cell_size, cell_size, cell_size), 
                               QPen(Qt.NoPen), QBrush(Qt.green))

        # Draw food
        self.scene.addRect(QRectF(self.game.food[1] * cell_size, self.game.food[0] * cell_size, cell_size, cell_size), 
                           QPen(Qt.NoPen), QBrush(Qt.red))

    def update_stats(self):
        stats = self.agent.get_stats()
        for stat, label in self.stats_labels.items():
            if stat == 'Action Distribution':
                actions = ['Straight', 'Right', 'Left']
                text = f"{stat}:\n" + "\n".join([f"{a}: {v:.2f}" for a, v in zip(actions, stats[stat])])
            else:
                value = stats[stat]
                text = f"{stat}: {value:.2f}" if isinstance(value, float) else f"{stat}: {value}"
            label.setText(text)

def train_and_visualize(episodes=10000, max_steps=1000):
    game = SnakeGame(width=20, height=15)
    agent = QLearningAgent(11, 3)
    
    app = QApplication(sys.argv)
    vis = SnakeVisualization(game, agent)
    vis.show()

    def train_step():
        nonlocal episodes
        if episodes > 0:
            state = game.reset()
            for step in range(max_steps):
                action_idx = agent.get_action(state)
                action = np.eye(3)[action_idx]
                next_state, reward, done = game.step(action)
                agent.update(state, action_idx, reward, next_state, done)
                state = next_state
                if done:
                    break
            
            agent.episode_count += 1
            agent.scores.append(game.score)
            agent.highest_score = max(agent.highest_score, game.score)
            agent.food_eaten += game.score
            agent.total_steps += game.steps
            agent.longest_snake = max(agent.longest_snake, len(game.snake))
            
            episodes -= 1
            if agent.episode_count % 100 == 0:
                print(f"Episode: {agent.episode_count}, Score: {game.score}, Epsilon: {agent.epsilon:.2f}")
        else:
            train_timer.stop()

    train_timer = QTimer()
    train_timer.timeout.connect(train_step)
    train_timer.start(5)  # Run as fast as possible

    sys.exit(app.exec_())

if __name__ == "__main__":
    train_and_visualize()