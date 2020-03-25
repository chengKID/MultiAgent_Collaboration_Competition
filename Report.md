## Report on Collaboration and Competition project



### Short description on Multi-Agent Deep Deterministic Policy Gradient 

This project is to train two agents whose action space is continuous using an reinforcement learning method called Multi-Agent Deep Deterministic Policy Gradient (MADDPG).  MADDPG is a kind of "Actor-Critic" method. Unlike DDPG algorithm which trains each agent independantly, MADDPG trains actors and critics using all agents information (actions and states). However, the trained agent model (actor) can make an inference independentaly using its own state.


In this report I briefly summarize the learnings and final model taken as part of the Collaboration and Competition project. After trying several different hyperparameters and models, I found a setup that solves the environment with 1202 episodes.



### Basic approach

Here I use Multi Agent Deep Deterministic Policy Gradients (MADDPG) in this project.


In this model every agent itself is modeled as a Deep Deterministic Policy Gradient (DDPG) agent where, however, some information is shared between the agents. In particular, each of the agents in this model has its own actor and critic model. The actors each receive as input the individual state (observations) of the agent and output a (two-dimensional) action. The critic model of each actor, however, receives the states and actions of all actors concatenated. This should facilitate the information sharing between the agents. The same as the last project where the actions were concatenated to the output of the first hidden layer. Throughout training the agents all use a common experience replay buffer (a set of stored previous 1-step experiences) and draw independent samples.

<img src="media/multi_agent_actor_critic.png" width="80%" align="center" alt="" title="Multi-Agent Actor-Critic" />

> _Figure 1: Multi-agent decentralized actor with centralized critic ([Lowe and Wu et al](https://papers.nips.cc/paper/7217-multi-agent-actor-critic-for-mixed-cooperative-competitive-environments.pdf))._



#### Network Architectures

**1. Actor network**

The actor network is a multi-layer perceptron (MLP) with 2 hidden layers, which maps states to actions.

   * Input Layer —> 1st hidden layer (256 neurons, ReLu) —> Batch Normalization —> 2nd hidden layer (128 neurons, ReLu) —> Output layer (tanh)

```py
class ActorModel(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, input_size, action_size, seed, fc1_units=256, fc2_units=128):
        super(ActorModel, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(input_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.bn1 = nn.BatchNorm1d(fc1_units)
        self.bn2 = nn.BatchNorm1d(fc2_units)
        self.reset_parameters()
        
    def forward(self, state):
        if state.dim() == 1:
            state = torch.unsqueeze(state,0)
        x = F.relu(self.fc1(state))
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))  
```

**2. Critic network**

The critic network is also a multi-layer perceptron (MLP) with 2 hidden layers, which maps (state, action) pair to Q-value.

```py
class CriticModel(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, input_size, actions_size, seed, fc1_units=256, fc2_units=128):
        super(CriticModel, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(input_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units+actions_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.bn1 = nn.BatchNorm1d(fc1_units)
        self.bn2 = nn.BatchNorm1d(fc2_units)
        self.reset_parameters()
        
    def forward(self, states, actions):
        """Build a critic network that maps (states, actions) pairs to Q-values."""
        xs = F.relu(self.fc1(states))
        xs = self.bn1(xs)
        x = torch.cat((xs, actions), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)
```


#### Training algorithm

1. The agents using the current policy and exploration noise interact with the environment, and the episodes are saved into the shared replay buffer.
2. For each agent, using a minibatch which is randomly selected from the reply buffer, the critic and the actor are trained using MADDPG training algorithm.
3. Target networks of actor and critic of agents are soft-updated respectively.

![MADDPG Algorithm](/media/maddpg_Algorithm.png)



##### Additional resource on DDPG
- [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275)


### Train two agents

After trying with different hyper-parameters and noise setting, the below set solves the environment and it is my final result.


#### Hyper parameters

In terms of hyperparameters I did a fair bit of experimenting. Here are the final choices and some
observations.

- Learning rate of the actor : 1e-4
- Learning rate of the critic : 1e-3
- Replay buffer size : 1e5
- Minibatch size : 256
- Discount factor : 1.0
- Tau : 1e-3
- Soft update of target parameter : 0.001


#### Result

The score increases very slowly in the early phase, and spikes after around 860th episode. After hitting the highest score above 2.7, it drops suddenly down to 0.1 or lower. Average score over 100 episodes reaches 0.513 at 1202 episode.

As outlined above the environment could be solved in 1202 episodes. The Progress is very slow to begin with, which is not unexpected due to the added noise in the process. The agent then stalls at an average score of ∼ 0.1 for more than 850 episodes. Only after that there is some dynamic in the training. The agent’s performance improves drastically starting at episode 860, then drops back down and rises with one short dip to reach average score 0.5 at episode 1202.


Another observation from looking at the individual episode scores: Even later on in training the performance is rather unstable. There are episodes with scores as high as 2.5 but also episodes with scores close to 0.


**Plot of reward**

![Scores](/media/plt_rewards.png)



### Ideas for Future work

Obviously fine tuning hyper-parameters is one of tasks to be done to stablize the policy to address the sudden drop of score after reaching the high score. For this, learning rate for actor and critic, and batch size are one to be tuned. Also, having different network architecture for actor and critic (deeper or wider) are something worth to be tried.


Another most obvious directions for additional research would be the change of the model. In particular, one could try Proximal Policy Optimization (PPO) on this task, which seems to have worked for other people in the nanodegree. I could imagine it to work quite well on a problem like this, that is not very high-dimensional.


There are, however, also potential improvements to be looked at within the approach I chose. The sampling from the buffer may be a crucial element of the model that could offer some levers of improvement. Another direction of work I imagine valuable is the choice and decay of noise introduced throughout training. I chose to stop noise after a certain number of training steps. But looking at better ways of initializing and gradually reducing the noise could be interesting.



### Addtional Resource

* [Scaling Multi-Agent Reinforcement Learning](https://bair.berkeley.edu/blog/2018/12/12/rllib/)
* [Paper Collection of Multi-Agent Reinforcement Learning (MARL)](https://github.com/LantaoYu/MARL-Papers)
