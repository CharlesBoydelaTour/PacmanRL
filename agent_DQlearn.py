import random
import pickle
from turtle import *
from environment import environment
from collections import namedtuple, deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(200, 50),
            nn.ReLU(),
            nn.Linear(50, 25),
            nn.ReLU(),
            nn.Linear(25, 4),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
    
class Agent(object):
    def __init__(self, env):
        self.env = env
        self.state = self.env.reset()
        self.reward = 0
        self.done = False
        self.action = [0,1,2,3] #['l','r','u','d']
        #self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.6
        self.count = 0
        self.total_reward = 0
        self.reward_history = []
        self.episodes = 0 
        self.batch_size = 32
        self.memory = ReplayMemory(10000)
        self.model = DQN().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.loss = nn.MSELoss()
        self.target_model = DQN().to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.target_update = 10
        self.target_count = 0
        self.target_update_count = 0
        self.epsilon_decay = 0.9
        self.epsilon_min = 0.2
        self.actionvonvert = {0:'l', 1:'r', 2:'u', 3:'d'}
        
    def get_action(self):
        if random.random() < self.epsilon:
            action = random.choice(self.action)
        else:
            #action = self.action[torch.argmax(self.model(torch.tensor(self.state).float().to(device))).item()]
            outputs = self.model(torch.tensor(self.state).float().to(device))
            action = self.action[np.random.choice(torch.where(outputs == outputs.max()))]
        return action
    
    def get_reward(self, action):
        next_state, reward, done = self.env.step(action)
        self.total_reward += reward
        self.reward = reward
        self.done = done
        self.state = next_state
        return next_state, reward, done
    
    def train(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        state_batch = torch.tensor(batch.state).float().to(device)
        action_batch = torch.tensor(batch.action).to(device)
        reward_batch = torch.tensor(batch.reward).float().to(device)
        next_state_batch = torch.tensor(batch.next_state).float().to(device)
        state_action_values = self.model(state_batch).gather(1, action_batch.unsqueeze(1))
        next_state_values = self.target_model(next_state_batch).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        loss = self.loss(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.target_count += 1
        if self.target_count % self.target_update == 0:
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_update_count += 1
            #print("Target Updated")
            #print("Target Updated Count: ", self.target_update_count)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            #print("Epsilon Updated")
            #print("Epsilon: ", self.epsilon)
        self.reward_history.append(self.total_reward)
        #print("Episode: ", self.episodes)
        #print("Total Reward: ", self.total_reward)
        #print("Average Reward: ", np.mean(self.reward_history))
        #print("Loss: ", loss)
        self.count += 1
        
    def save_model(self):
        torch.save(self.model.state_dict(), 'model.pth')
        print("Model Saved")
    def load_model(self):
        self.model.load_state_dict(torch.load('model.pth'))
        print("Model Loaded")
        
    def run(self):
        for i in range(100):
            self.state = self.env.reset()
            self.done = False
            nb_move = 0
            while not self.done:
                action = self.get_action()
                next_state, reward, self.done = self.get_reward(self.actionvonvert[action])
                self.memory.push(self.state, action, next_state, reward)
                self.train()
                self.state = next_state
                nb_move += 1
                if nb_move > 1000: self.done = True
            self.save_model()
            self.reward_history.append(self.total_reward)
            self.episodes += 1
            print("Episode: ", self.episodes)
            print("Total Reward: ", self.total_reward)
            print("Average Reward: ", np.mean(self.reward_history))
            self.total_reward = 0
            self.count += 1
    
if __name__ == '__main__':    
    setup(420, 420, 370, 0)
    hideturtle()
    tracer(False)
    env = environment()
    agent = Agent(env)
    Q, rhist = agent.run()
    
    with open('qvalues.pickle', 'wb') as handle:
        pickle.dump(Q, handle, protocol=pickle.HIGHEST_PROTOCOL)
