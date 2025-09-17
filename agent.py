import torch
import random
import numpy as np
from snake_game import SnakeGame
from model import Linear_QNet
from trainer import QTrainer
from collections import deque
import os

MAX_MEMORY = 100_000
BATCH_SIZE = 1000

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # Exploration rate
        self.gamma = 0.95 # Discount rate
        self.model = Linear_QNet(11, 256, 3)
        self.memory = deque(maxlen=MAX_MEMORY)
        self.trainer = QTrainer(self.model, lr=0.001, gamma=self.gamma)

        if os.path.exists('./model/model.pth'):
            self.load_checkpoint('./model/model.pth')
            print("Checkpoint loaded! Resuming training.")
        else:
            print("No checkpoint found. Starting new training.")

    def get_state(self, game):
        head = game.snake[0]
        
        # Define points around the head
        point_l = (head[0], head[1] - 1)
        point_r = (head[0], head[1] + 1)
        point_u = (head[0] - 1, head[1])
        point_d = (head[0] + 1, head[1])

        points = [point_u, point_r, point_d, point_l]  # clockwise: Up, Right, Down, Left

        # Get current direction as booleans
        dir_l = game.direction == (0, -1)
        dir_r = game.direction == (0, 1)
        dir_u = game.direction == (-1, 0)
        dir_d = game.direction == (1, 0)

        directions = [dir_u, dir_r, dir_d, dir_l]  # clockwise: Up, Right, Down, Left

        # Rotate points to match current direction
        for i in range(4):
            if directions[i]:
                points = points[i:] + points[:i]
                break

        state = [
            # Danger straight
            points[0] in game.snake or not (0 <= points[0][0] < game.GRID_SIZE and 0 <= points[0][1] < game.GRID_SIZE),

            # Danger right
            points[1] in game.snake or not (0 <= points[1][0] < game.GRID_SIZE and 0 <= points[1][1] < game.GRID_SIZE),

            # Danger left
            points[3] in game.snake or not (0 <= points[3][0] < game.GRID_SIZE and 0 <= points[3][1] < game.GRID_SIZE),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food[0] < head[0],  # Food is Up
            game.food[0] > head[0],  # Food is Down
            game.food[1] < head[1],  # Food is Left
            game.food[1] > head[1]   # Food is Right
        ]

        return np.array(state, dtype=int)

    def get_action(self, state):
        self.epsilon = max(0.01, 0.3 - self.n_games / 400) # Example of a slower decay
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state_tensor = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state_tensor)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        
        return final_move


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory
        
        # Unzip the mini_sample
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        # We will pass this to our trainer
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        # We will also add a method to train on just the last move
        self.trainer.train_step(state, action, reward, next_state, done)


    def load_checkpoint(self, filepath):
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.n_games = checkpoint['n_games']
        # Set model to training mode
        self.model.train()

    # --- NEW: Checkpoint saving method ---
    def save_checkpoint(self, filepath):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.trainer.optimizer.state_dict(),
            'n_games': self.n_games
        }
        torch.save(checkpoint, filepath)
