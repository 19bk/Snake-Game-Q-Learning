import numpy as np
import random
from collections import deque
import pygame
import time


# Optimized LearnSnake class
class LearnSnake:
    def __init__(self, width=20, height=15):
        self.width = width
        self.height = height
        self.reset()

    def reset(self):
        self.snake = deque([(self.height // 2, self.width // 2)])
        self.direction = (0, 1)
        self.food = self._place_food()
        self.score = 0
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
        
        dir_l, dir_r, dir_u, dir_d = self.direction == (0, -1), self.direction == (0, 1), self.direction == (-1, 0), self.direction == (1, 0)

        state = (
            (dir_r and point_r in self.snake) or (dir_l and point_l in self.snake) or 
            (dir_u and point_u in self.snake) or (dir_d and point_d in self.snake),
            (dir_u and point_r in self.snake) or (dir_d and point_l in self.snake) or 
            (dir_l and point_u in self.snake) or (dir_r and point_d in self.snake),
            (dir_d and point_r in self.snake) or (dir_u and point_l in self.snake) or 
            (dir_r and point_u in self.snake) or (dir_l and point_d in self.snake),
            dir_l, dir_r, dir_u, dir_d,
            self.food[1] < head[1], self.food[1] > head[1],
            self.food[0] < head[0], self.food[0] > head[0]
        )

        return state

    def step(self, action):
        # [straight, right, left]
        clock_wise = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        idx = clock_wise.index(self.direction)
        
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            new_dir = clock_wise[(idx + 1) % 4]
        else:  # [0, 0, 1]
            new_dir = clock_wise[(idx - 1) % 4]

        self.direction = new_dir
        new_head = ((self.snake[0][0] + self.direction[0]) % self.height, 
                    (self.snake[0][1] + self.direction[1]) % self.width)

        self.snake.appendleft(new_head)
        
        reward = 0
        self.game_over = new_head in list(self.snake)[1:]
        if self.game_over:
            reward = -10
        elif new_head == self.food:
            self.score += 1
            reward = 10
            self.food = self._place_food()
        else:
            self.snake.pop()
        
        return self._get_state(), reward, self.game_over

# Optimized SnakeQAgent class
class SnakeQAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = 0.01
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.q_table = {}

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        return np.argmax(self.q_table.get(state, np.zeros(self.action_size)))

    def update(self, state, action, reward, next_state, done):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_size)
        q_value = self.q_table[state][action]
        next_max = np.max(self.q_table.get(next_state, np.zeros(self.action_size)))
        new_q = q_value + self.learning_rate * (reward + self.gamma * next_max - q_value)
        self.q_table[state][action] = new_q

        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train(episodes=10000, max_steps=1000):
    env = LearnSnake()
    agent = SnakeQAgent(11, 3)
    scores = []
    
    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        
        for step in range(max_steps):
            action = agent.get_action(state)
            action_vector = np.eye(3)[action]
            next_state, reward, done = env.step(action_vector)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        scores.append(total_reward)
        
        if e % 100 == 0:
            print(f"Episode: {e}, Score: {total_reward}, Epsilon: {agent.epsilon:.2f}")
    
    return scores, agent

def play_game(agent, num_games=5):
    env = LearnSnake()
    pygame.init()
    screen = pygame.display.set_mode((env.width * 20, env.height * 20))
    clock = pygame.time.Clock()

    for game in range(num_games):
        state = env.reset()
        done = False
        score = 0
        
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            action = agent.get_action(state)
            action_vector = np.eye(3)[action]
            state, reward, done = env.step(action_vector)
            score += reward

            # Visualize the game
            screen.fill((0, 0, 0))
            for segment in env.snake:
                pygame.draw.rect(screen, (0, 255, 0), (segment[1] * 20, segment[0] * 20, 20, 20))
            pygame.draw.rect(screen, (255, 0, 0), (env.food[1] * 20, env.food[0] * 20, 20, 20))
            pygame.display.flip()
            clock.tick(10)  # Control game speed

        print(f"Game {game + 1} Score: {score}")
        time.sleep(1)  # Pause between games

    pygame.quit()

if __name__ == "__main__":
    start_time = time.time()
    print("Starting training...")
    scores, trained_agent = train()
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds")
    
    print("Starting game play with trained agent...")
    play_game(trained_agent)