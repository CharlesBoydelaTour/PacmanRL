<p align="center">
  <a href="https://upload.wikimedia.org/wikipedia/commons/2/26/Pacman_HD.png" rel="noopener">
 <img width=200px height=200px src="https://upload.wikimedia.org/wikipedia/commons/2/26/Pacman_HD.png" alt="Project logo"></a>
</p>

<h3 align="center">Pac-Man Reinforcement Learning: a Q-Learning Agent in different environments</h3>

---

<p align="center"> The purpose of this project is to implement a RL environment in python of the Game Pac-Man, in order to train a Q-Learning agent, and study its behaviours with respect to variation in the environments.
    <br> 
</p>

## 📝 Table of Contents

- [About](#about)
- [Getting Started](#getting_started)

## 🧐 Introduction `<a name = "about"></a>`

Pac-Man is a well-known game which was first released in Japan on May 22, 1980. The game was then
released in North America in October 1980. Pac-Man is a maze game where the player controls the titular character, Pac-Man, as he attempts to eat all the pellets in a maze. The game is over when all of the pellets have been eaten or if Pac-Man is caught by one of the four ghosts.

Many Reinforcement Learning approaches have been taken to create game achieving good results. The most recent algorithms are a variant of the Q-learning (Deep-QLearning) algorithm and uses a neural network to estimate the value for each state. These methods scan the game image as input and learn the Q function by a deep convolutional neural network. Applied to many Atari games by the [OpenAI gym platform](https://gym.openai.com/docs/), these methods give good results. As can be shown in the graph below (*[Mnih, Volodymyr, et al. &#34;Human-level control through deep reinforcement learning.&#34; nature 518.7540 (2015): 529-533](https://deepmind.com/blog/article/deep-reinforcement-learning).*), the PacMan game still has very low scores compared to other Atari games.

Thus, our work seeks to reimplement a simplified version of the game to be able to analyse the behaviour of an agent in front of various evolutions of the environment to understand what can be the blocking points during the learning process.

<p align="center">
  <a href="https://lh3.googleusercontent.com/TMRwiE8uKLAzUA3UQ7yXOaDmjAvkWU_25Z7Bnic04QLxE1pPV9dYurLnnSTfZqbF-0E11zrzkeXAYRD-s5jWN97DIuKs7T1-t30dJQ=w2048-rw-v1" rel="noopener">
 <img width=400px height=700px src="https://lh3.googleusercontent.com/TMRwiE8uKLAzUA3UQ7yXOaDmjAvkWU_25Z7Bnic04QLxE1pPV9dYurLnnSTfZqbF-0E11zrzkeXAYRD-s5jWN97DIuKs7T1-t30dJQ=w2048-rw-v1" alt="Results Atari"></a>
</p>

## 🏁 Getting Started `<a name = "getting_started"></a>`

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See [deployment](#deployment) for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them.

```
Give examples
```

### Installing

A step by step series of examples that tell you how to get a development env running.

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo.

## 🔧 Running the tests `<a name = "tests"></a>`

Explain how to run the automated tests for this system.

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## 🎈 Usage `<a name="usage"></a>`

Add notes about how to use the system.

## 🚀 Deployment `<a name = "deployment"></a>`

Add additional notes about how to deploy this on a live system.

## ⛏️ Built Using `<a name = "built_using"></a>`

- [MongoDB](https://www.mongodb.com/) - Database
- [Express](https://expressjs.com/) - Server Framework
- [VueJs](https://vuejs.org/) - Web Framework
- [NodeJs](https://nodejs.org/en/) - Server Environment

## ✍️ Authors `<a name = "authors"></a>`

- Charles Boy de la Tour
- Megan Fillion

## 🎉 Acknowledgements `<a name = "acknowledgement"></a>`

- Mnih, Volodymyr, et al. "Human-level control through deep reinforcement learning." *nature* 518.7540 (2015): 529-533.
- Mnih, Volodymyr, et al. "Playing atari with deep reinforcement learning." *arXiv preprint arXiv:1312.5602* (2013).
- Gnanasekaran, Abeynaya, Jordi Feliu Faba, and Jing An. "Reinforcement learning in pacman." See also URL http://cs229. stanford. edu/proj2017/final-reports/5241109. pdf (2017).
- Qu, Shuhui, Tian Tan, and Zhihao Zheng. "Reinforcement Learning With Deeping Learning in Pacman."
