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
