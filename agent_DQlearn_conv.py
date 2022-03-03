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

    def __init__(self, h = 9, w = 20, outputs = 4):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5, stride=2)
        #self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, stride=2)
        #self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=5, stride=2)
        #self.bn3 = nn.BatchNorm2d(16)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return self.head(x.view(x.size(0), -1))
    
class Agent(object):
    def __init__(self, env):
        self.env = env
        self.state = self.env.reset()
        self.state = np.array(self.state).reshape((20,int(len(self.state)/20)))
        self.reward = 0
        self.done = False
        self.action = [0,1,2,3] #['l','r','u','d']
        self.alpha = 0.1
        self.gamma = 0.8
        self.epsilon = 0.8
        self.count = 0
        self.total_reward = 0
        self.reward_history = []
        self.episodes = 0 
        self.batch_size = 32
        self.memory = ReplayMemory(1000)
        self.model = DQN().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss = nn.MSELoss()
        self.target_model = DQN().to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.target_update = 10
        self.target_count = 0
        self.target_update_count = 0
        self.epsilon_decay = 0.9999
        self.epsilon_min = 0.01
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
        next_state = np.array(next_state).reshape((20,int(len(next_state)/20)))
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
        state_batch = state_batch.view(self.batch_size,1,20,int(len(state_batch[0])/20))
        next_state_batch = next_state_batch.view(self.batch_size,1,20,int(len(next_state_batch[0])/20))
        print(state_batch.shape)
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
        for i in range(1000):
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
