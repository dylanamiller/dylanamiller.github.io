---
title: Actor Critics
author: Dylan Miller
layout: post
---
This is my third post, and I am already super bored with writing the "Introduction" section. I'm keeping this brief:

Back in 2015, the Machine Learning (ML) community was all abuzz with the news that a small company out of England named Deepmind - since aquired by Google - was able to train a [Reinforcement Learning](https://en.wikipedia.org/wiki/Reinforcement_learning) (RL) algorithm (or agent) to achieve human and super human performace in many [Atari](https://www.nature.com/articles/nature14236) games. They accomplished this by using an artificial neural network (ANN) as the function approximator for what was effectively a [q-learning](https://dylanamiller.github.io/2019/12/17/temporal-difference-learning.html) algorithm. This prompted a veritable explosion of RL research, which included further work on Deep-Q Network (DQN, Deepmind's algorithm) style algorithms as well as developments of various [policy gradient](https://dylanamiller.github.io/2019/12/25/policy-gradients.html) (PG) methods.

I am going to assume that if you happened to find this post, you already know something about RL; if you are not sure what all of the above means, the two posts I linked to above and the Wikipedia page will give you a good idea. If you do not want to look at those, I will give a super fast recap of RL, TD, and PG. After this quick detour, we will pick up where the above left off.

## **Ultra-Quick Reinforcement Learning Recap**

![](/assets/images/actor_critic/rl_loop.png "RL Closed Loop")

If you are planning to do any work in RL, familiarize yourself with this image. What it is describing is a [Markov Decision Process](https://en.wikipedia.org/wiki/Markov_decision_process) (MDP) that is being used to define a RL problem. A MDP is the mathematical formulation typically used to define RL problems. It consists of a set of states that the agent can potentially find itself in, the set of actions that the agent can take that move it from state to state, the transition function that probabilistically (or deterministically) defines how the agent moves from one state to another, and the reward function that gives signal to the agent in response to its movement - I know this is not entirely complete, but it will serve our purposes. Through this formulation, the agent interacts with an environment, collectiong experience as it moves from state to state in an attempt to learn how to maximize some notion of a reward. The agent chooses its actions by means of a policy, the optimization of which corresponds the maximization of reward. In this post we will look at model-free methods. This means that the agent has no knowledge of the transition or reward functions; this is in contrast to model-based methods in which the agent has this knowledge, even if it may not be perfect.

One of the important things to note about the above, is that unlike in supervised learning, the reward that the agent receives is evaluative, not instructive. So, the agent is only ever getting a sense for how good its action was, not whether or not it was the best action it could have taken.

Q-learning, mentioned above, is a [temporal difference](https://cling.csd.uwo.ca/cs346a/extra/tdgammon.pdf) method. These algorithms use value funcitions, which act as a proxy to the agent's policy that I mentioned above. Rather than explicitly tell the agent how to adjust its policy given its experience, value methods estimate how much cumulative reward can be expected given a state or state-action pair and guide the agent to update these estimates, and therefore behavior, according to whether or not their actions lead to better results. TD algorithms are bootstrapping. This means that they are able to perform their updates at each time step.

The other class (broadly speaking) of RL algorithm is the [policy gradient](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf) method. Unlike value methods, PG algorithms apply their updates in an attempt to directly optimize the policy. There are several theoretical advantages to this (refer to my policy gradient post, linked above) as well as some drawbacks that will be mentioned in the next section. PG methods perform Monte Carlo updates, as in [REINFORCE](http://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf), which means that they collect full trajectories before performing an optimzation step.

Whew, glad that's over. Moving on...

## **Where We Left Off**

Before blacking out briefly, I was saying that there have been significant advances in both PG and TD algorithms. Despite this, each class of algorithms has its own set of inherent problems. In particular, each is plagued by a problem that is the complement of the other: as TD methods bootstrap, their updates introduce significant bias, whereas PG methods use a Monte Carlo update which has zero bias, but high variance. If only there existed some way to tie the two together and reap the benefit of both.....oh, right. That's what actor-critics do.

So, before we dive in, let's real quick correct a couple of ideas that some people (maybe you, maybe not) have about actor-critics. First, actor-critic is not an algorithm. Is is an architecture, consisting of an actor and a critic, that enables certain behavior in an agent. An algorithm that uses this architecture to enforce this behavior is an actor-critic algorithm. Second, is that not all actor-critics are PG algorithms. It is perfectly possible to design an actor-critic that uses only value functions to define behavior. This works in the other direction as well: not all PG algorithms are actor-critics (see the link to REINFORCE above). 

## **Actor-Critics**

How does an actor-critic operate such that it gets the best of both TD and PG?

Each component of the algorithm serves a difference purpose. At a high level, the actor chooses and action given the current state, and the critic judges how good this action was compared to its own estimation. It took me a number of times seeing this sort of usesless description of the mechanics before I was able to process what was actually going on. It also helps to understand a bit about the classes of algorithms being being to create this framework. 

At any rate, what the above description means practically, is that the agent chooses an action from whatever policy is being used. Following this, the critic, who does not take part in choosing the action and instead keeps a running estimate of the value function, calculates the one step TD error, shown in the equation below, to establish the quality of that action in light of the current value function estimate. By having useful one step information (i.e. the TD error), the agent is able to update both components of the algorithm in a bootstrapping way, rather than have to wait for the full trajectory to perform a Monte Carlo update.

![](/assets/images/actor_critic/td_error.png "TD error")

Note: the TD error that the critic uses is the TD error in the true definition. Q-learning uses an update that looks very similar, but with the action values rather than the state values. This is not the true TD error, though people often refer both as the TD error.

Another Note: There are many algorithms that have an actor and a critic, but could be argued against being actor-critic algorithms. According to [Sutton and Barto](http://www.incompleteideas.net/book/bookdraft2017nov5.pdf) on page 274, in the first paragraph of the section "Actor-Critic Methods," it is stated that if the value function estimated by the critic is used only as a baseline (see [REINFORCE with baseline](https://dylanamiller.github.io/2019/12/25/policy-gradients.html)), and it is not used to enable bootstrapping, then it is not considered an actor-critic. 

The TD error calulated by the critic, is fed to the actor - there are many instances of algorithms that people call actor-critics that do not do this, but this is how the framework was originally defined -
and becomes part of its update. The resulting updates are:

![](/assets/images/actor_critic/critic_update.png "Critic Update")

and

![](/assets/images/actor_critic/actor_update.png "Actor Update")

Though they look very similar, the behavior resulting from each is intuitively different. THe critic, updating based on the state value, moves toward the optimal value function, while the actor, even though it also uses the TD error, this can be seen as the TD error that is a funciton of the state and action, moving toward the optimal policy - unless of course it is not a PG actor, in which case it moves toward the optimal action value function. 

All of this leads to an MDP that can now be illustrated as follows:

![](/assets/images/actor_critic/ac_loop.png "AC Loop")

The state comes from the environment to each the actor and the critic. The actor selects an action, transitions to a new state, and the reward is fed to the critic, which calculates the TD error and passes it to the actor.

## **Implementation**

Actor-critic methods were originally all on-policy algorithms, which means that the experience used to perform updates belongs to the policy that is being used to choose actions. This is in contrast to off-policy algorithms, where the experience does not have to be from the action choosing policy - this has to do with what the algorithm is directly trying to optimize for and how that effects the stationarity (I am well aware that this is likely not a word, but I like it and it gets my point across) of the of the parameter space. While this is beyond the scope of this post, I'll give a quick example, since I'm the one writing and I feel like it - plus it's Friday night and I have no date, so time is of no issue at the moment. 

Quick Example: Q-leaning is off-policy as it is trying to estimate the optimal action value function. So it collects data from on policy (eg. epsilon-greedy) and makes updates toward a different one. Sarsa is structured [very similarly](https://dylanamiller.github.io/2019/12/17/temporal-difference-learning.html) to q-learning, but is on-policy. It chooses actions from a policy and makes updates toward that policy. Really both of these update value functions, but the action decisions are affected as a result. So, q-learning is trying to estimate a stationary object, the optimal value function, toward which all trajectories converge. Sarsa on the other hand, is trying to improve the estimate for a non-stationary object, the action choosing policy, for which older trajectory data may no longer be relevent. 

Note: Actor-critics now have the ability to made off-policy. How? Because Richard Sutton is a [bamf](https://arxiv.org/pdf/1205.4839.pdf), thats how.

If you have some knowledge of TD and PG algoritm structure - information to which has been linked to several times by now - then it is fairly easy to implement a simple actor-critic algorithm given the above two updates and the equation for TD error. Pseudocode is below.

![](/assets/images/actor_critic/ac_algorithm.png "AC Algorithm")

Full code can be found on my [github](https://github.com/dylanamiller/actor_critic). Among the files is an actor-critic that uses its critic to bootstrap by feeding the TD error to the actor, as well as an "actor-critic" that is really more of a REINFORCE with a baseline.


