<p align="center">
  <a href="https://upload.wikimedia.org/wikipedia/commons/2/26/Pacman_HD.png" rel="noopener">
 <img width=200px height=200px src="https://upload.wikimedia.org/wikipedia/commons/2/26/Pacman_HD.png" alt="Project logo"></a>
</p>

<h3 align="center">Pac-Man Reinforcement Learning: a Q-Learning Agent in different environments</h3>

---

<p align="center"> The purpose of this project is to implement a RL environment in python of the Game Pac-Man, in order to train a Q-Learning agent, and study its behaviour with respect to variation in the environments.
    <br> 
</p>

## 📝 Table of Contents

- [REPORT](#report)
  - [I.Introduction and motivation](#iintroduction-and-motivation)
  - [II.The environment](#iithe-environment)
  - [III.The agent](#iiithe-agent)
  - [IV.Discussion and results](#ivdiscussion-and-results)
- [Additional](#additional)
  - [🏁 Getting Started](#-getting-started)
  - [🔧 Running tests](#-running-tests)
  - [✍️ Authors](#️-authors)
  - [🎉 Acknowledgements](#-acknowledgements)
# REPORT

## I.Introduction and motivation

Pac-Man is a well-known game which was first released in Japan on May 22, 1980. The game was then
released in North America in October 1980. Pac-Man is a maze game where the player controls the titular character, Pac-Man, as he attempts to eat all the pellets in a maze. The game is over when all of the pellets have been eaten or if Pac-Man is caught by one of the four ghosts.

Many Reinforcement Learning approaches have been taken to create game achieving good results. The most recent algorithms are a variant of the Q-learning (Deep-QLearning) algorithm and uses a neural network to estimate the value for each state. These methods scan the game image as input and learn the Q function by a deep convolutional neural network. Applied to many Atari games by the [OpenAI gym platform](https://gym.openai.com/docs/), these methods give good results. As can be shown in the graph below (*[Mnih, Volodymyr, et al. &#34;Human-level control through deep reinforcement learning.&#34; nature 518.7540 (2015): 529-533](https://deepmind.com/blog/article/deep-reinforcement-learning).*), the PacMan game still has very low scores compared to other Atari games.

<p align="center">
  <a href="https://lh3.googleusercontent.com/TMRwiE8uKLAzUA3UQ7yXOaDmjAvkWU_25Z7Bnic04QLxE1pPV9dYurLnnSTfZqbF-0E11zrzkeXAYRD-s5jWN97DIuKs7T1-t30dJQ=w2048-rw-v1" rel="noopener">
 <img width=400px height=700px src="https://lh3.googleusercontent.com/TMRwiE8uKLAzUA3UQ7yXOaDmjAvkWU_25Z7Bnic04QLxE1pPV9dYurLnnSTfZqbF-0E11zrzkeXAYRD-s5jWN97DIuKs7T1-t30dJQ=w2048-rw-v1" alt="Results Atari"></a>
</p>

Thus, our work seeks to reimplement a simplified version of the game to be able to analyse the behaviour of an agent in front of various evolutions of the environment to understand what can be the blocking points during the learning process.

## II.The environment

We decided to reimplement a PacMan model to reduce the size of the maps. 2 different maps have been used: a simple map (circular 15x7), and a complex map (which just add a line cuting the first map in 2).
In the standart configuration one ghost starts near the player. It can be removed.
The reward is 100 if the player moves through all the squares without being "eaten" by a ghost. It also get a reward of 10 if it goes on a case where it has not been before.
If the ghost eat the player (they are on the same position), the player receive a negative reward of -100.
In addition to the original rules, we have added a negative score of -10 if the player performs an impossible action (hits a wall). If the game is not finished after 500 moves, the game is over.

Our first approach was to provide the status of the set of boxes in the game as the state of our environment for our agent.This approach was quickly reduced to speed up the learning process.

Thus we considered two states of the environment:

- In the first case, the agent receives as information the relative distance (horizontal and vertical) from the nearest unvisited square, and the relative distance from the nearest ghost. This approach drastically reduces the number of possible states to be visited, but seems to be a good compromise regarding the relevance of the information available to the player.
- In the second case, we have reduced the state of the environment to the boxes directly accessible by our agent. As we will see in our analysis, this leads to good results and fast learning, but the agent detects possible ghosts at the last moment which leaves it little space to act.

## III.The agent

The agent is Pacman. Its action space is {left, right, up, down}. To chose its action at each turn, it follows a epsilong-greedy policy : the agent choose the action with the maximum expected retun with a probability (1−𝜖) and a random action to explore the action space with a probability 𝜖 (here 0.1).

The agent learns the action-value function through Q-Learning. A state-action pairs table Q initialised at 0 is then updated at each moves with the following formula.

<p align="center">
  <a href="https://wikimedia.org/api/rest_v1/media/math/render/svg/678cb558a9d59c33ef4810c9618baf34a9577686" rel="noopener">
 <img width=500px height=100px src="https://wikimedia.org/api/rest_v1/media/math/render/svg/678cb558a9d59c33ef4810c9618baf34a9577686" alt="Q-Learning Formula"></a>
</p>

It is an off-policy temporal difference control algorithm which approximates the optimal action-value function of the system.

## IV.Discussion and results

# Additional

## 🏁 Getting Started

Python 3.9.7

The following libraries have to be installed:

- turtle
- collections
- numpy
- matplotlib

## 🔧 Running tests

To train the Agent, just run Agent_Qlearn_closevision.py or Agent_Qlearn_statedistance.py
During training, reduce the turtle window to disable the rendering.
To change the map, go in environment.py and change the self.tilesinit attribute.

## ✍️ Authors

- Charles Boy de la Tour
- Megan Fillion

## 🎉 Acknowledgements

- Mnih, Volodymyr, et al. "Human-level control through deep reinforcement learning." *nature* 518.7540 (2015): 529-533.
- Mnih, Volodymyr, et al. "Playing atari with deep reinforcement learning." *arXiv preprint arXiv:1312.5602* (2013).
- Gnanasekaran, Abeynaya, Jordi Feliu Faba, and Jing An. "Reinforcement learning in pacman." See also URL http://cs229.stanford.edu/proj2017/final-reports/5241109.pdf (2017).
- Qu, Shuhui, Tian Tan, and Zhihao Zheng. "Reinforcement Learning With Deeping Learning in Pacman."
