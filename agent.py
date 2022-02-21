import torch
import random
from collections import deque
from game import SnakeGameAI, Direction, Point, BLOCK_SIZE
from model import SnakeAIModel, SnakeAITrainer
from helper import plot
import numpy as np

LR = 0.001
MAX_MEMORY = 100000
BATCH_SIZE = 1000

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomnes
        self.gamma = 0.8 # for loss function calculation
        self.memory = deque(maxlen=MAX_MEMORY) # pop from left
        self.model = SnakeAIModel(11, 256, 3)
        self.trainer = SnakeAITrainer(gamma=self.gamma, lr=LR, model=self.model)

    def get_state(self, game):
        # [Danger straight, Danger right, Danger left,  Direction Left, Direction Right, Direction Up, Direction Down,  Food Left, Food Right, Food Up, Food Down]
        
        head = game.snake[0]
        # Create arbitary points
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        # Check Direction
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # Create State
        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),

            # Directions
            dir_l, dir_r, dir_u, dir_d,

            # Food left
            game.food.x < head.x,
            # Food Right
            game.food.x > head.x,
            # Food Up
            game.food.y < head.y,
            # Food Down
            game.food.y > head.y 
        ]   

        return np.array(state, dtype=int)

    def get_action(self, state):
        final_move = [0, 0, 0]
        self.epsilon = 100 - self.n_games
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            predicted = self.model(state0)
            index = torch.argmax(predicted).item()
            final_move[index] = 1
        return final_move
        

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        # Check if memory is less than batch size
        if len(self.memory) < BATCH_SIZE:
            mini_sample = self.memory
        else:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.train_short_memory(states, actions, rewards, next_states, dones)

    def train_short_memory(self, states, actions, rewards, next_states, dones):
        self.trainer.train_step(states, actions, rewards, next_states, dones)        

def train():
    plot_scores = []
    plot_mean_scores = []
    best_score = 0
    total_score = 0
    
    # Initializing the objects
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # Get the agent state
        state = agent.get_state(game)

        # Get the action / move
        action = agent.get_action(state)

        reward, done, score = game.play_step(action)
        # Get new state
        new_state = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state, action, reward, new_state, done)
        # Remember
        agent.remember(state, action, reward, new_state, done)

        if done:
            # Reset the game
            game.reset()

            # train long memory, plot result
            agent.train_long_memory()
            agent.n_games += 1

            if score > best_score:
                best_score = score
                agent.model.save()

            print(f'Games : {agent.n_games}, Score : {score}, Best Score : {best_score}')
            plot_scores.append(score)
            total_score += score
            plot_mean_scores.append(total_score / agent.n_games)
            plot(plot_scores, plot_mean_scores)


if __name__ == "__main__":
    train()