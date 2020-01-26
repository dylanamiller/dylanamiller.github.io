---
title: TD Learning
subtitle: Temporal Difference Learning
author: Dylan Miller
layout: post
---
Reinforcement learning (RL) is a field that has received a great deal of attention since 2015, when some large-headed individuals seemed to come from the woods declaring that they had figured out how to get the top score at the arcade. [Or something...](https://www.nature.com/articles/nature14236) \*cough.\* Either way, RL, and especially the Deep-Q Network (DQN) algorithm, have since proven to be able to solve problems of a different type of complexity than other machine learning (ML) methods. These problems are not ones that can be modeled statically, but rather require control over a series of decisions that correspond to some action at that point in time, in order to be properly tackled. The aim of these decisions, and thereby actions, is to maximize the total amount of some reward that is is received by the algorithm, or agent, over time. The correct actions needed to achieve this reward, are determined through what is essentially a trial and error process. 

A good example of this process is one from basic [classical conditioning](https://en.wikipedia.org/wiki/Classical_conditioning). Imagine there is a mouse in a cage (do not worry, no mice were harmed in the making of this example, so keep reading...darn hippie). The experimenter makes accessible to the mouse two levers. At first, the mouse does nothing. After several times of this happening, maybe the mouse moves toward the levers. At this point, the experimenter drops a food pellet close to the levers so that the mouse moves closer to them. This repeats several times, at which point the mouse decides to push one of the levers. OH NO! The mouse got shocked! After recovering from this traumatic experience, the experimenter continues placing food close to the levers. But this time the mouse is wiser, and does not push the same lever, moving instead for the other one. The mouse pushes it and is delighted to be rewarded with more food. In further experimentation, the mouse now knows what to do and which lever to press to get all the food it can eat. (This same experiment was actually done, not by me, with cocaine also. Needless to say, the mice got highly addicted to cocaine and died shortly thereafter, because the only reward they wanted was the cocaine, and so they forget to eat.)

This behavior was learned purely as a result of the mouse receiving positive rewards, the food, for actions that it took which moved it toward some desirable goal state, and negative rewards, the shock, for actions which did not.

Note: Despite my wording in the above example’s conclusion, the notion of a “goal state” is not defined in RL. Although frequently used, this idea is simply one to make the explanation less abstract and a bit easier to grasp for people new to the concept. In fact, it is not even necessary that the series of decisions ever end, let alone end in a specific “goal state.”

So, now it is probably time that I present the pieces that do define RL, since I have been so audacious as to claim that certain things do not.

In simplest terms, RL is an area of ML in which an agent attempts to learn to navigate an environment based on interactions with this environment in which the agent will receive some reward in response to the actions it takes ([Sutton and Barto, 1998](http://incompleteideas.net/book/bookdraft2017nov5.pdf)). The agent accomplishes this through the closed loop process described by the image below, which if you have ever seen RL material, and I am being liberal in my use of the word material, you have seen:

![](/assets/images/td_learning/rl_loop.png "RL Closed Loop")

It is worth noting, that one of the key pieces differentiating the RL problem from a Supervised Learning problem, is the reward piece of the above loop. In RL, this signal is evaluative, not instructive. In other words, the agent receives information telling it how good the action it took was, not whether or not it was the best action it could have taken. 

As complicated as it may get, this image is at the core of all RL work. But, while good for a high-level understanding, this image is not a formal description that can be used in algorithm development. Let’s define a mathematical framework that we can use for that purpose.

## **Markov Decision Process**

Markov Decision Processes (MDP) are one way to think about the structure of a RL problem. Although there are multiple ways to define the MDP, for simplicity we will call ours a tuple consisting of a set of states, a set of action, a transition function, and a reward function, where the states are all possible “positions” that the agent can find itself in at some time step, the actions are all possible things the agent can do, the transition function is the stochastic or deterministic function describing movement from one state to another – although there is RL work done in which the transition function is known, model-based RL, what we will focus on here is model-free RL, which assumes the agent does not know the transition function – and the reward function describes what rewards the agent will get for given state-action pairs, which like the transition function, our agent will now know.

Armed now with the MDP, we can describe the loop shown in the image above as such: an agent observes the state it is in, takes some action available to it, moves to another state given the environment’s unknown dynamics, and receives a reward. This reward will be used by the agent to determine how good the action taken was given the state it was in. The agent will then update its decision making scheme accordingly.

And what exactly is this decision making scheme? It seems like it would be important to understand. This scheme is called the policy. The policy is an either deterministic or stochastic mapping (function) of states to actions that ultimately decides the agents course of action. The policy can be improved directly or, as will be the subject for the remainder of this post, through the updating of a proxy function. The latter is the case in Temporal Difference Learning.

## **Temporal Difference Learning**

Temporal Difference (TD) learning is a method for policy evaluation that uses bootstrapping to improve its representation of the environment’s dynamics. So, unlike Monte Carlo based methods that need to sample entire trajectories of experience before performing an update, TD methods improve their estimation of the underlying policy at each step using the existing estimation of the dynamics. While TD learning is not inherently an algorithm for control, with minor tweaks to the math this can be changed. These methods to tweak TD learning for control include Q-learning and Sarsa, the two algorithms I will be walking through later in this post. 

But, before we get there (ha! And you thought we were jumping into the algorithms) there are a few things we should cover.

### *Value Functions and Bellman Equations*

As previously mentioned, TD methods use a proxy function to the policy rather than the policy itself. This proxy is called a value function. The first value function is the state-value function 

![](/assets/images/td_learning/state_value_fcn.png "State Value Fcn")

This function tells you what return (i.e. cumulative reward) the agent can expect to get given the state it is currently in and following the current policy. The second value function is the action-value function

![](/assets/images/td_learning/action_value_fcn.png "Action Value Fcn")

While similar to the state-value function, the action value function instead tells you what return the agent can expect if if were to take some action given the current state, and thereafter follow the current policy. This is important, because compared to the state-value function, it tells you the effect of diverging from the policy in the given state. So, if our goal is control, it may be worth remembering that the action-value function deals in the effect of taking a certain action, possibly contrary to the policy.

In their current forms, these value functions do not help us much, but an earlier piece of information I let slip should point us in the right direction. We want to be able to perform updates at each step, so we need a way to extract information describing the performance of our agent as it moves from one state to the next. 

This can be done through the Bellman equations for v, q. The Bellman equations provide a way to recursively show the expected return for the current state. In other words, they enable us to describe the value functions in terms of the current state’s reward and the expected return for another state. In this case, that other state is the next state, whatever it may be. Given this, our value functions become

![](/assets/images/td_learning/v_bellman.png "Bellman Eqn for v")

![](/assets/images/td_learning/q_bellman.png "Bellman Eqn for q")

These will enable us to write our TD update (the td-error) as

![](/assets/images/td_learning/td_error.png "TD error")

We use Q for this, because our goal is control. Otherwise we are simply performing a policy evaluation, which tells us the quality of the policy underlying our estimation of the value function, but does not allow us to take actions or perform updates. Also, ![](/assets/images/td_learning/gamma.png "Discount") is a discount factor between 0 and 1 used to decrease future rewards and thereby lessening their importance compared to more immediate ones.

Intuitively, what does the td-error tell us? In english, it says that given my current state, how much reward did I receive for take the action that moved me to the next state, add to that the return expected for the next state, and then compare this to the return that the current value function estimate says to expect given the current state. This results in the difference between what was expected to happen and what actually happened. So, if our td-error is positive, then the actual reward was more than what was expected and we can increase our value function estimate for that state-action pair. If it is negative, then the actual reward received was less than expected and we can decrease our value function estimate for that state, action pair.

## **Implementation Details**

We are almost there. What is left are a couple topics that are important to RL, but can be handled in ways other than what we will discuss.

### *Exploration vs Exploitation*

In order for our agent to be successful, it must properly trade-off between trying new actions in states that may yield higher return and using the knowledge its experience has already taught it about the environment. This is the exploration versus exploitation problem. There are different ways to handle it, but the one we will use is called an epsilon-greedy policy. What this rule says, is that given some ![](/assets/images/td_learning/epsilon.png "Epsilon"), with probability 1-epsilon, choose the action that has the highest value when the current state and all of the possible actions are processed by the q function. So, choose the action with the highest expected return. This seems like it would be good idea in general. No? 

The problem with this, is that since the reward our agent receives is evaluative, not instructive, having the highest q value does not ensure that an action actually yields the best results. Only that it is so far a reasonably good choice. Since the agent does not know the underlying dynamics of the environment, it could in fact be the worst possible action. If the agent always acts greedily, it will never learn the difference.

In order to avoid this, epsilon-greedy also says that with probability epsilon, the agent chooses an action at random. With enough experience, this ensures a roughly equal sampling of all possible actions in all possible states. This is called greedy in the limit with infinite exploration, or GLIE.

Since over time, it is assumed that the agent is improving, the epsilon value can be decayed. With more experience, the value function estimation approaches the real value function, the the greedy action has a higher probability of being the actual best action.

### *Function Approximation*

With all its treatment as something special, DQN is essentially just Q-learning that uses an artificial neural network as its function approximator. It is not anything particularly exotic – all respects of course to the people at Deepmind who do phenomenal work. 

But because of this, we will not be implementing DQN, but regular Q-learning. What! *Gasp* ML without neural networks?! That’s right. Think of it as my way of sticking it to the man. Also though, I think hype for neural networks has obscured a lot of what RL actually is. And although I definitely respect and appreciate their use as function approximators, I want this post to be more about the fundamental aspects of TD algorithms, not how to correctly use neural networks.

What will we use instead? Is there anything else? The time before neural networks was so long ago, I can barely remember. I was not even sure I existed until Facebook used CNNs to pick my face out of a crowd. 

Fear not, there was a before.

We will be using a [Fourier basis](https://people.cs.umass.edu/~pthomas/papers/Konidaris2011a.pdf) for our function approximator. A Fourier series approximates a function by combining different frequency signals in order to create a function that matches the desired signal to some arbitrary complexity. Obviously the higher the order (i.e. the number of frequency signals) the closer the approximation will be, but for many problems a relatively low order will do. The Fourier basis can be written as 

![](/assets/images/td_learning/fourier_basis.png "Fourier Basis")

Where, since we will use a univariate expansion, ![](/assets/images/ci.png) is an integer that will be multiplied by ![](/assets/images/td_learning/pi.png) and the state to create feature i. The representation with contain a total of order + 1 features. The features are then added together and combined with the weight parameters in order to find the q-values for a given action.

## **Q-learning and Sarsa**

And we made it. Given all that we have so far covered, the actual algorithms for Q-learning and Sarsa should be relatively straight forward, so this section will be brief. They are listed below and a complete implementation of each can be found on my [github](https://github.com/dylanamiller/temporal-difference-learning). 

#### Q-learning:
![](/assets/images/td_learning/q_learning.png "Q-learning Algorithm")

#### Sarsa:
![](/assets/images/td_learning/sarsa.png "Sarsa Algorithm")

For both algorithms, d0 is the initial state distribution and the q functions are parameterized by the weights w. a' and s' indicate the next action and state respectively and r is the reward.

These two algorithms are very similar, differing in that Q-learning is an off-policy algorithm while Sarsa is on-policy. The reason for this, is that despite both algorithms using an epsilon-greedy policy to choose the action in the current state, Q-learning chooses the max action when in the next state, whereas Sarsa continues to follow the epsilon-greedy policy. This greedy action selection means that Q-learning is attempting to use the optimal policy, which is different than the policy it is trying to improve, for action selection, which makese it an off-policy algorithm.

note: More information about Q-learning and Sarsa can be found in [Sutton and Barto, 1998](http://incompleteideas.net/book/bookdraft2017nov5.pdf), 
