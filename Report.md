# MAFS5370 Assignment 2 - Derivative Hedging and Pricing Based on Reinforcement Learning

- SHAO, Ruizhao 
- WANG, Liangshu

## Problem Statement
Assuming binary tree price distribution, we are proposing an MDP hedging model for ATM American options with 10 steps maturity
## 1. Introduction
In general, pricing American options is cumbersome. To our knowledge, a purely analytical solution to impermanent American options does not exist. Consequently, obtaining an analytical, explicit, and a non-cumbersome solution seemed impossible. 
Several methods were suggested to solve the problem of pricing American options using the Black–Scholes framework with considerable limitations, such as the lack of simplicity and accuracy. 
We proposed a hybrid method with the benefits of both analytical and numerical methods to start with a standart MDP formation and policy gradient algorithm to decide the optimal execution strategy. ATM American pricing could be extracted from the above optimal execution policy with numerical simulation

## 2. Background
Policy-based MDP methods search for a policy directly, rather than searching for a value function and extracting a policy. This is similar to Q-learning and SARSA we deployed in our assignment1, but instead of learning a Q-function, we are updating the parameter set $\theta$ directly using gradient descent. 

In policy gradient methods, we approximate the policy from the rewards and actions received in our episodes, similar to the way we do it with Q-learning. We can do this provided that the policy has two properties:

1. The policy is represented using some function that is differentiable with respect to its parameters. For a non-differentiable policy, we cannot calculate the gradient.

2. Typically, we want the policy to be stochastic. Recall from the section on policies that a stochastic policy specifies a probability distribution over actions, defining the probability with which each action should be chosen.

The goal of a policy gradient is to approximate the optimal policy 
 via gradient ascent on the expected return. Gradient ascent will find the best parameters 
 for the particular MDP.

## 3. MDP Formulation
Variables we used in this model are summarized as follows:
|Variable|Description|Value|
|:-:|:-:|:-:|
|$S_0$|Initial Stock Price|100|
|$S_t$|Stock Price Development|Float|
|$K$|Strike Price|100|
|$R_f$|Risk Free Rate|5%|
|$\sigma$|volatility|0.2|
|$T$|Time to Maturity|10|
## 4. Methodology

## 5. Results

## 6. Discussion
Comparing with SARSA/Q-learning technic we used in Assignment1, the major advantage of policy-gradient approaches is that they can handle high-dimensional action and state spaces, including actions and states that are continuous. This is because we do not have to iterate over all actions using $argmax_{\alpha \in A(S)}$ as we do in value-based approaches. For continuous problems, $argmax_{\alpha \in A(S)}$ is not possible to calculate, while for a high number of actions, the computational complexity is dependent on the number of actions.

However, a disadvantage of REINFORCE is known as sample inefficiency. Since the policy gradients algorithm takes an entire episode to do the update, it is difficult to determine which of the state-action pairs are those that effect the value G(the episode reward).

Moreover, model-free reinforcement learning is a particularly challenging case to understand and explain why a policy is making a decision. This is largely due to the model-free property: there are no action definitions that can used as these are unknown. However, policy gradients are particularly difficult because the values of states are unknown: we just have a resulting policy. With value-based approaches, knowing $V$ or $Q$ provides some insight into why actions are chosen by a policy; although explainability problems still remain.
