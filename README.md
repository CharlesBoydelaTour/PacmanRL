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
    - [Simple Maze state distance approach](#simple-maze-state-distance-approach)
    - [Simple Maze close view approach](#simple-maze-close-view-approach)
    - [Complex Maze state distance approach](#complex-maze-state-distance-approach)
    - [Complex Maze close view approach](#complex-maze-close-view-approach)
  - [V.Conclusion and next steps](#vconclusion-and-next-steps)
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

### Simple Maze state distance approach

<p align="center">
  <a href="/Simple%20Maze/State%20distance/simplemaze-distances.gif" rel="noopener">
 <img width=300px height=200px src="./Simple%20Maze/State%20distance/noghost/simplemaze-distances-noghost.gif" alt="GAME"></a>
</p>

Our first implementation is a simple maze with no ghost. As can be seen above on the final game after 200 training games, the player seems to be stucked when it reaches a position which is at an equal distance from 2 closest unseen positions. This lead to an unpredictable behaviour of the agent, which at some points during the learning, finished the game without any difficulties, whereas it tends to break afterwards.

<p align="center">
  <a href="/Simple%20Maze/State%20distance/simplemaze-distances.gif" rel="noopener">
 <img width=300px height=200px src="./Simple%20Maze/State%20distance/simplemaze-distances.gif" alt="GAME"></a>
</p>

As can be seen in our second impletentation, adding a ghost tends to "unblock" our agent. Indeed, the agent learns quickly to escape from the ghost when it gets too close.

<p align="center">
  <a href="/Simple%20Maze/State%20distance/simplemaze-distances.gif" rel="noopener">
 <img width=320px height=240px src="./Simple%20Maze/State%20distance/simplifiedmap-200steps.png" alt="Reward"></a>
</p>

The graph above reprents the total rewards over each trained game for the simple maze with state distance approach agent. We can see that after 75 games, the reward fluctuates. The agent is still dependent on the movements of the ghost which may lead to unseen states. However, it achieved the maximum number of points/rewards (500) more and more frequently as we increase the training time.

### Simple Maze close view approach

<p align="center">
  <a href="/Simple%20Maze/State%20distance/simplemaze-distances.gif" rel="noopener">
 <img width=300px height=200px src="./Simple%20Maze/State%20Close%20Vision/simplemaze-closevision.gif" alt="GAME"></a>
</p>

In the close view approach, the agent as only access to the values of the nearby blocks. It did not get stuck as the first state, but since the player only see the ghosts when it is at one block, it as only a small room for maneuver.

<p align="center">
  <a href="/Simple%20Maze/State%20distance/simplemaze-distances.gif" rel="noopener">
 <img width=320px height=240px src="./Simple%20Maze/State%20Close%20Vision/Figure_1.png" alt="Reward"></a>
</p>

As can be seen on the total reward other epoch graph above, the agent still achieves max scores, and do so faster than the first approach.

### Complex Maze state distance approach

In the complex maze example, we are adding on line to the maze (note the ghost can not go in this line). This leads to some additional complexity since now our agent can wait indefinitely in this line without the fear of being eaten by the ghost.

<p align="center">
  <a href="/Simple%20Maze/State%20distance/simplemaze-distances.gif" rel="noopener">
 <img width=300px height=200px src="./Complex%20Maze/State%20distance/complexmaze-distances.gif" alt="GAME"></a>
</p>

For the state distance approach, this leads to similar results where there is a high number of unvisited states, and the agent tends to get stuck, but with a good understanding of the ghost risk.

<p align="center">
  <a href="/Simple%20Maze/State%20distance/simplemaze-distances.gif" rel="noopener">
 <img width=320px height=240px src="./Complex%20Maze/State%20distance/complexmap-800steps.png" alt="Reward"></a>
</p>
From the reward graph, we can see that a higher number of training is required to achieve similar results, with only a few games reaching the maximum number of points.

### Complex Maze close view approach

<p align="center">
  <a href="/Simple%20Maze/State%20distance/simplemaze-distances.gif" rel="noopener">
 <img width=300px height=200px src="./Complex%20Maze/State%20Close%20Vision/complexmaze-closevision.gif" alt="GAME"></a>
</p>

With the complex approach, the agent has the same difficulties as in the simple maze. If the ghost get to close, the available window to fly away is really small.

However, it seems to have a better scaling compare to the other approach. Indeed, looking at the reward graph, we can see that the agent reaches the maximum number of points quite often.

<p align="center">
  <a href="/Simple%20Maze/State%20distance/simplemaze-distances.gif" rel="noopener">
 <img width=320px height=240px src="./Complex%20Maze/State%20Close%20Vision/Figure_1.png" alt="Reward"></a>
</p>

## V.Conclusion and next steps

In conclusion, due to the size of the possible states, especially with a high number of ghosts, solving the PacMan game is not an easy RL task.
A good representation of these states is a key to achieve an efficient training for the agent.
We have tried two of these representations : minimal distances, and close view.
These two approaches led to really different results, with their own advantages and limits.

- Minimal distances seems to get stuck at some positions, probably due to the high number of states to explore and ambiguity between the distances. It had a lot of difficulties to scale to a maze with increasing complexity.
- Close view achieved a faster training and a better scaling in increaing complexity but had issues in escaping the ghost.

To go further, a combination of these 2 methods, with for example the absolute distances and the close view could improve the performance of the agent.
Other algorithm such as SARSA could also be used and could lead to better performance during the training process.

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
