[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/42135622-e55fb586-7d12-11e8-8a54-3c31da15a90a.gif "Soccer"

# Multi-Agent Project: Collaboration and Competition

## Train Two RL Agents to Play Tennis


### Project Background: Why Multi-agent RL Matters
For artificial intelligence (AI) to reach its full potential, AI systems need to interact safely and efficiently with humans, as well as other agents. There are already environments where this happens on a daily basis, such as the stock market. And there are future applications that will rely on productive agent-human interactions, such as self-driving cars and other autonomous vehicles.

One step along this path is to train AI agents to interact with other agents in both cooperative and competitive settings. Reinforcement learning (RL) is a subfield of AI that's shown promise. However, thus far, much of RL's success has been in single agent domains, where building models that predict the behavior of other actors is unnecessary. As a result, traditional RL approaches (such as Q-Learning) are not well-suited for the complexity that accompanies environments where multiple agents are continuously interacting and evolving their policies.


### Introduction

For this project, we will work with the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment and solve it using RLdeep learning based models for multi-agent continuous controls and actions.

![Trained Agent][image1]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 


### Solving the environment

The task is episodic, and in order to solve the environment, the agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 consecutive episodes) of those **scores** is at least +0.5.


### Setting up the environment

1. The environment can be downloaded from one of the links below for all operating systems:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    - _For AWS_: To train the agent on AWS (without [enabled virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment.  The agent can **not** be watched without a virtual screen, but can be trained.  (_To watch the agent, one can follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

2. Place the downloaded file in the same directory as this GitHub repository and unzip the file.

3. Use the `requirements.txt` file to set up a python environment with all necessary packages installed.


### Instructions

* `Tennis.ipynb`
  * A jupyter notebook where provides introduction to the environment and follow all the steps to create RL environment, train the RL and test the environment.
* `ddpg_agent.py`
  * A module which define the MADDPG training algorithm
  * A module which defines the DDPG Agent class, ReplayBuffer
  * The agents will be trained with MADDPG alogorithm
* `model.py`
  * Deep neural networks for actor and critic are defined


### Approach and solution

The reinforcement learning approach we use in this project is called Multi Agent Deep Deterministic Policy Gradients (MADDPG). see this [paper](https://papers.nips.cc/paper/7217-multi-agent-actor-critic-for-mixed-cooperative-competitive-environments.pdf). In this model every agent itself is modeled as a Deep Deterministic Policy Gradient (DDPG) agent (see this [paper](https://arxiv.org/pdf/1509.02971.pdf)) where, however, some information is shared between the agents.

In particular, each of the agents in this model has its own actor and critic model. The actors each receive as input the individual state (observations) of the agent and output a (two-dimensional) action. The critic model of each actor, however, receives the states and actions of all actors concatenated.

Throughout training the agents all use a common experience replay buffer (a set of stored previous 1-step experiences) and draw independent samples.

Details of the implementation including the neural nets to model actor and critic models can be found in the modules `maddpg_agent.py` and `models.py` as well as the report (`Report.md`). With the current set of models and hyperparameters the environment can be solved well reach the average score 0.513 in around 1202 episodes.