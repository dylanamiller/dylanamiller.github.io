---
title: Policy Gradients
author: Dylan Miller
layout: post
---
Reinforcement Learning (RL) has received a lot of love in the past few years. This is largely due to Deepmind’s release of the [Deep Q-Network](https://www.nature.com/articles/nature14236) (DQN) paper in Nature back in 2015. In the paper, researchers used an Artificial Neural Network (ANN) as the nonlinear function approximator for a q-learning algorithm (with some added bells and whistles), allowing the RL algorithm, or agent, to play and win a number of Atari games, thereby achieving a major milestone in recent RL research – games serve as a good benchmark for RL algorithms since they are complex enough to demonstrate complex learned behavior, but remain in the realm of a well-defined problem. And, since adding an ANN to something automatically makes it cool (because...[SCIENCE](https://github.com/dylanamiller/reinforce/blob/master/science/scientific_verification.md). Caution: the above does not hold, as I have since found, in the case of face tattoos), RL has become a prime actor in the field of Artificial Intelligence (AI) and Machine Learning (ML), only slightly shadowed by Convolutional Neural Networks (CNN) and other algorithms that have helped elegantly solve our severe cat classification problem.

Despite its waltz into the limelight, DQN is only one of a number of RL algorithms that have managed to achieve some impressive results. DQN is a member of the class of [Temporal Difference](https://dylanamiller.github.io/2019/12/17/temporal-difference-learning.html) (TD) methods, which sits at variable distance from the equally prominent algorithms that fall into the Policy Gradient (PG) family - while DQN seems to have become synonymous with the class of algorithm to which it belongs, PG seems to have become the accepted term to refer to all methods that belong to it. As you may have guessed (titles have the tendency to spoil surprises), this post will give an introduction into PG methods. In the final section, we will go over the REINFORCE algorithm [Williams, 1992](http://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf).

But first, since this is an RL post, not just a PG post, the obligatory “What is RL?” section. Do not fret, I will be brief. If I am too brief ([Sutton and Barto, 1998](http://incompleteideas.net/book/bookdraft2017nov5.pdf)) is free and is full of excellent and in depth information.

## **Reinforcement Learning: A quick overview**

Any person who has seen RL material has likely come across the image:

![](/assets/images/policy_gradients/rl_loop.png "RL Closed Loop")

What the above image is describing is the Markov Decision Process (MDP) that is the base for much of the work being done in RL. Although a very high level view, the above image successfully shows that an agent exists in some state and interacts with its environment by means of performing an action. In response to this, the environment yields to the agent a reward and the agent moves to some other state according to the dynamics (which may be deterministic or stochastic) of the environment. For our purposes, because we are discussing what is called model-free RL, as opposed to model-based RL, our agent does not know what these dynamics are or what the reward function is – if we were looking at model-based approaches there would be an additional element of planning involved in our definition of the problem. The agent selects actions via a learned policy, and the goal of the agent is to learn the policy (or surrogate function to the policy) that maximizes the total cumulative, possibly discounted, reward. 

These described components, the set of states, the set of actions, the reward function, the transition function, and additionally the initial state distribution are what define our MDP. This gives us our formal mathematical framework for describing problems as RL problems.

note: There are other ways an RL problem can be defined, but MDPs are the most common method.

An important aspect of the above description that helps differ RL from supervised learning, is that the reward that the agent receives is evaluative as opposed to instructive. This means that unlike in supervised learning, where there is a label attached to a data sample that gives an explicit right or wrong that can be optimized for by way of a loss function, in RL the agent only gets a relative sense for how it is doing; the optimal behavior is not explicitly communicated.

## **Policy Gradients**

With our platform established, we can now look at what PG algorithms are and what they do. 

In contrast to TD methods, PG algorithms do not use a value function as a surrogate for the policy, but rather elect to optimize a parameterized policy directly. We can define as our object functions

![](/assets/images/policy_gradients/pg_objective.png "PG Objective")

and perform gradient ascent with respect to the policy parameters in order to maximize the expected return, rather than gradient descent to minimize the error as one would in supervised learning. We perform our updates to the parameters as

![](/assets/images/policy_gradients/pg_update.png "PG Update")

### *Advantages of Policy Gradients*

There a few reason in particular why we may want to choose PG algorithms over TD ones (it’s okay, I don’t think DQN will get jealous). 

1. For starters, by having a parameterized policy we can easily account for a continuous action space. Rather than needing to determine which action to take by discrete q-values, the parameterization allows a continuum of values more readily. 

2. In addition, because we do not rely on q-values, the algorithms are more robust to slight changes in action probability. Epsilon greedy is a common policy method in q-learning that frequently chooses the action with the highest q-value, which means that slight changes to the q-values can lead to drastically different behavior. This can make stochastic more difficult to correctly represent.

3. Since PG are, as the name would suggest, a class of gradient algorithms, they tend to have better convergence properties than TD algorithms. A good example of this appears in q-learning (or any off policy algorithm), where [Tsitsiklis, et. al. 1997](https://www.mit.edu/~jnt/Papers/J063-97-bvr-td.pdf) proved that using a nonlinear function approximator for these types of algorithms can lead to divergence. Yes, this includes DQN since ANNs act as a nonlinear function approximator.

This is not a complete list of PG advantages (again, see Sutton and Barto), but it conveys a proper sense for why we may like them.

### *Policy Gradient Theorem*

The issue then becomes, how do we accurately estimate the gradient of our objective function without knowledge of the transition or reward functions. And, how do we do it in a way that leverages the underlying structure of the problem (i.e. that we are using an MDP to formulate our problem). 

Thankfully, [Sutton, et al., 2000](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf) derived an analytic solution to these questions. The policy gradient theorem states 

![](/assets/images/policy_gradients/policy_gradient_theorem.png "Policy Gradient Theorem")

In the above equation, q(s,a) is the q function for state s and action a and ![](/assets/images/policy_gradients/partial.png) is the partial derivative of the policy at the pair s, a, with respect to its current parameters. This gives us the direction the parameters are required to change in order to increase the probability of the actions that yield higher returns and decrease the probability of the actions that yield lower returns. 

note: The above equation holds for parameters of the current policy. This is why typically PG algorithms are considered on-policy, meaning that the policy choosing the agents actions is the same as the policy that is being updated – this differs from an algorithm like q-learning, which chooses actions by attempting to estimate the optimal policy. Despite this, off-policy PG algorithms do exist.

![](/assets/images/policy_gradients/d_theta.png) can be seen intuitively as a probability distribution over states, although this is not entirely correct. It is not a valid probability distribution, because it does not sum to 1. This can be change by multiplying the term by (1-![](/assets/images/policy_gradients/gamma.png)). What the term ends up doing, is averaging over the state distributions at different time steps, weighting more heavily the earlier ones. This cause ![](/assets/images/policy_gradients/j_theta.png) to favor updates to the parameters that increase the expectation of return at earlier time steps.

note: While initially proven for finite MDPs (MDPs that have a guaranteed stopping point) and deterministic initial state, it holds for continuous MDPs and non-deterministic initial states.

### *Temporal Difference Algorithms are not Gradient Algorithms*

Before moving onto the REINFORCE algorithm, I want to discuss briefly why the TD update found in q-learning – or the popular DQN – is not a gradient update. First, lets recall the td-error term that defines the update:

![](/assets/images/policy_gradients/td_error.png "TD error")

Here, v(.) is the state-value function, which describes the return that can be expected if the policy is followed from the current state – this is contrary to the action-value function, q(s,a), which describes the return that can be expect if a certain action is taken and thereafter the policy is followed. Intuitively, this is describing how to perform the update based on whether what actually happened was better or worse that what was expected to happen. If we define this to be our loss function, we get

![](/assets/images/policy_gradients/td_loss.png "TD loss")

Then, lets take the derivative with respect to the weights:

![](/assets/images/policy_gradients/td_grad.png "TD gradient")

This is differs from the TD update, because it will both increase ![](/assets/images/policy_gradients/v_s.png) and decrease ![](/assets/images/policy_gradients/v_splus.png), whereas the TD update will only increase ![](/assets/images/policy_gradients/v_s.png). Additionally, it is not an unbiased update (see [Baird, 1995](http://leemon.com/papers/1995b.pdf)).

## **REINFORCE: Vanilla Policy Gradient**

REINFORCE, also known as the Vanilla Policy Gradient, is a Monte Carlo PG algorithm. So, we will perform our updates at the end of each episode, or after one trajectory comes to its completion. We will not be able to use the policy gradient theorem in its above form, but with a few manipulations it will be ready for implementation.

To start, let’s rewrite the theorem:

![](/assets/images/policy_gradients/policy_gradient_theorem.png "Policy Gradient Theorem")

Now remember from differential calculus that

![](/assets/images/policy_gradients/ln_deriv.png "Derivative of log")

Using this, we can write the theorem as

![](/assets/images/policy_gradients/pgt_new.png "PGT new form")

Which then becomes

![](/assets/images/policy_gradients/pgt_final.png "PGT final form")

This will be the form we use in REINFORCE. The pseudo code for the algorithm is below.

![](/assets/images/policy_gradients/reinforce.png "REINFORCE")

Where G is the episode's return. Look to my [github](https://github.com/dylanamiller/reinforce) for a full implementation.

As you can see, the algorithm is relatively straight forward. Collect a trajectory and perform an update, and there is not much more to it than that. This algorithm works fairly well despite its simplicity (maybe because of it – I tend to like simplicity), but it is subject to a common pitfall of Monte Carlo methods: it has high variance. Luckily, there is a way we can reduce this variance to increase training stability and therefore improve training time – there are actually several ways, but we will focus on one.

### *Baselines*

Given that our gradient is an expectation, we can make changes to the value as long as the expectation remains intact. This can be done by using a baseline that is subtracted off of the q-value that appears in the policy gradient theorem, as long as the baseline is not dependent on the action. While there are several ways in which this can be done, a common one is to use an estimate of the state-value function, which depends only on the state. This gives us

![](/assets/images/policy_gradients/reinforce_baseline.png "REINFORCE Baseline")

as our updated algorithm. In this I change G to q, because since we are using the state-value function as our baseline, which can be estimated on a step-by-step basis, we can write the return as the action-value function, or q, because the return will correspong to the actions we actually took.

note: q(s,a) - v(s) is the advantage function, which tells us how much better it is to take an action in a given state than the one that the policy dictates.

