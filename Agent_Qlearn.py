import random
import pickle
from turtle import *
from environment import environment
import numpy as np

class Agent(object):
    def __init__(self, env):
        self.env = env
        self.state = self.env.reset()
        self.modedifiedstate()
        self.reward = 0
        self.done = False
        self.action = ['l','r','u','d']
        self.Q = {}
        self.alpha = 0.1
        self.gamma = 0.8
        self.epsilon = 0.2
        self.count = 0
        self.total_reward = 0
        self.reward_history = []
        self.episodes = 0   
        
    def modedifiedstate(self):
        self.statemod = np.array(self.state)
        self.statemod = self.statemod.reshape((int(len(self.state)/20), 20))
        y_player, x_player = np.where(self.statemod == 3)
        #find coordinates of cois (where value is 1)
        y_coin, x_coin = np.where(self.statemod == 1)
        #find closest coin to player
        if len(x_coin) > 0:
            coin_dist = [abs(x_player[0] - x_coin[i]) + abs(y_player[0] - y_coin[i]) for i in range(len(x_coin))]
            closest_coin = np.argmin(coin_dist)
            dx_coin = x_player - x_coin[closest_coin]
            dy_coin = y_player - y_coin[closest_coin]
            self.state = (dx_coin[0], dy_coin[0])
        else : self.state = (0, 0)
        
    def get_action(self):
        if random.uniform(0,1) < self.epsilon:
            return random.choice(['l','r','u','d'])
        else:
            return self.choose_best_action()
    
    def choose_best_action(self):
        max_q = -float('inf')
        best_action = None
        for action in self.action:
            q = self.Q.get((self.state, action))
            if q is None : q = 0
            if q >= max_q:
                max_q = q
                best_action = action
        return best_action
    
    def update_q_table(self, action, reward, next_state):
        q = self.Q.get((self.state, action))
        if q is None:
            self.Q[(self.state, action)] = reward
        else:
            self.Q[(self.state, action)] = q + self.alpha * (reward + self.gamma * self.get_max_q(next_state) - q)
            
    def get_max_q(self, state):
        max_q = -float('inf')
        for action in self.action:
            q = self.Q.get((state, action), 0)
            if q >= max_q:
                max_q = q
        return max_q
    
    def run(self):
        for train in range(100):
            while True:
                action = self.get_action()
                next_state, reward, done = self.env.step(action)
                self.update_q_table(action, reward, next_state)
                self.state = next_state
                self.modedifiedstate() #
                self.total_reward += reward
                self.count += 1
                if self.count >= 500: done = True
                if done:
                    self.episodes += 1
                    self.reward_history.append(self.total_reward)
                    self.epsilon = 1 / self.episodes
                    self.total_reward = 0
                    self.state = self.env.reset()
                    self.modedifiedstate() #
                    self.count = 0
                    done = False
                    break
        return self.reward_history, self.Q
    
if __name__ == '__main__':    
    setup(420, 420, 370, 0)
    hideturtle()
    tracer(False)
    env = environment()
    agent = Agent(env)
    rewaesn, Q = agent.run()
    
    #with open('qvalues.pickle', 'wb') as handle:
    #    pickle.dump(Q, handle, protocol=pickle.HIGHEST_PROTOCOL)
