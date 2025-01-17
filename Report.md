## MAFS5370 Assignment 2 - Derivative Hedging and Pricing Based on Reinforcement Learning

- SHAO, Ruizhao 
- WANG, Liangshu

### Problem Statement
In a binomial model of a single stock with non-zero interest rate, assume that we can hedge any fraction of a stock, use policy gradient to train the optimal policy of hedging an ATM American put option with maturity T = 10. When do you early exercise the option? Is your solution same as what you obtain from delta hedging? **[DONE]** 

**Substitute** : For the really advanced students:  suppose the stock follows a GBM, construct an algorithm to train a NN that hedges an ATM American put option. **[DONE]** 

**Optional bonus** : after solving the above optional question, use the Soft Actor Critic algorithm. **[DONE]** 

**Advanced Extra bonus** : implement the GAC algorithm to solve the optional problem. **[Partially DONE]**

### 1. Introduction
In general, pricing/hedging American options is cumbersome. To our knowledge, a purely analytical solution to impermanent American options does not exist. Consequently, obtaining an analytical, explicit, and a non-cumbersome solution seemed impossible. 
Several methods were suggested to solve the problem of pricing American options using the Black–Scholes framework with considerable limitations, such as the lack of simplicity and accuracy. 
We proposed a hybrid method with the benefits of both analytical and numerical methods to start with **a standard MDP formation and Policy Gradient algorithm to decide the optimal hedging strategy of an ATM American put option as well as the exercising point.** ATM American pricing could be extracted from the above optimal policy with numerical simulation.

In this report, we tried several methods to complete the policy gradient method under different task settings, they are:

1. **Non-Network PG Agent under Binomial Tree setting**
2. **Network PG Agent under GBM setting**
3. **Deep Deterministic Policy Gradient Agent under GBM setting**
4. **Soft Actor-Critic Agent under GBM setting**
5. **Guide Actor-Critic Agent under GBM setting (NOT Finished)**

### 2. Background
Policy-based MDP methods search for a policy directly, rather than searching for a value function and extracting a policy. This is similar to Q-learning and SARSA we deployed in our assignment1, but instead of learning a Q-function, we are updating the parameter set $\theta$ directly using gradient descent. 

In policy gradient methods, we approximate the policy from the rewards and actions received in our episodes, similar to the way we do it with Q-learning. We can do this provided that the policy has two properties:

1. The policy is represented using some function that is differentiable with respect to its parameters. For a non-differentiable policy, we cannot calculate the gradient.

2. Typically, we want the policy to be stochastic. Recall from the section on policies that a stochastic policy specifies a probability distribution over actions, defining the probability with which each action should be chosen.

The goal of a policy gradient is to approximate the optimal policy via gradient ascent on the expected return. Gradient ascent will find the best parameters for the particular MDP.

### 3. Non-Network PG Agent under Binomial Tree setting
#### 3.1. MDP Formulation

Environment Variables we used in this model are summarized as follows:

|Variable|Description|
|:-:|:-:|
|$S_0$|Initial Stock Price|
|$S_t$|Stock Price Development|
|$K$|Strike Price|
|$R_f$|Risk Free Rate|
|$\sigma$|volatility|
|$T$|Time to Maturity|
The stock prices follow Binomial Tree development. 

**States (Finite):** The state is a combination of time and current stock price. 

**Actions (Finite):** The actions are the hedge ratio we choose to hedge the American put option, in this setting, we define it as finite, like (0, 0.05, 0.1, 0.15, 0.2, ..., 0.95, 1.0), 21 actions in total.

**Reward:** The opposite of the absolute value of the hedging portfolio value change

**Policy:** Actions are chosen based on the policy function directly.

#### 3.2 Policy Gradient Algorithm

We conduct the policy gradient algorithm in the following way:

1. Initialize the policy value for each state and action tuple.

2. Based on the Binomial Tree model of stock, generate path of a stock price movement.
3. Based on the stock price and the option price in the movement, choose the action using the existing policy function (values), calculate the reward and update the policy value using gradient ascent. 
4. We also decide whether to exercise the option during the process: If the current payoff of exercising the option is larger than the current option price, we will choose to exercise the option.
5. Check the convergence and test the model.

#### 3.3 Results

1. **Convergence Test:**

   The following chart is the reward convergence chart, the y-axis is the reward value plus 20 (20 is added to the original reward (strictly negative) to make the reward able to be positive. Then we can update the probability to both sides, which is good for the algo to converge)

   ![](D:\box\HKUST\23spring\5370\convergence_1.png)

2. **Comparison with delta hedging:**

   The following chart shows the comparison between the agent decision (Hedge Ratio) and the delta hedging value (Delta Hedge) in one random stock path:

   ![image-20230508190621069](C:\Users\patrick\AppData\Roaming\Typora\typora-user-images\image-20230508190621069.png)

   Overall, the agent can give a hedge ratio close to delta hedging, which proves our algorithm is working. If we look deep into it, we can find that when the delta value are relatively large (0.19, 0.23, 0.35) or relatively small (2.5*10^(-15), 0), the agent can find a hedge ratio very close to delta hedging. However, when the delta value are close to zero and small (0.01, 0.03, 0.08), there is still errors between them.

3. **Early Exercise Test:**

   We choose two examples to show the early exercise decision of our agent:

   - No exercise:

   <img src="C:\Users\patrick\AppData\Roaming\Typora\typora-user-images\image-20230508185135573.png" alt="image-20230508185135573" style="zoom:50%;" />

   - ​	Early exercise:

   <img src="C:\Users\patrick\AppData\Roaming\Typora\typora-user-images\image-20230508185217332.png" alt="image-20230508185217332" style="zoom:50%;" />

   We can see that when the stock price continue to go up, we will not exercise the American put option for sure. And if the stock price goes down and it happens in a early stage, we will have the chance to exercise the option, which is same as our experience about American put option.

### 4. Network PG Agent under GBM setting

#### 4.1. MDP Formulation

Environment Variables in the GBM setting are similar to the Binomial, the only difference is we add number of steps to make the steps larger than 10 (we set 50). This will make the path longer and force the agent to learn more.

### 5. Deep Deterministic Policy Gradient Agent under GBM setting

Deep Deterministic Policy Gradient (DDPG) is a type of reinforcement learning algorithm that combines both deep neural networks and policy gradients to solve complex decision-making problems in continuous action spaces.

In the context of option pricing, DDPG can be used to learn an optimal policy for an agent to determine the optimal exercise strategy for an American option. Specifically, the agent can use the DDPG algorithm to learn the optimal policy for determining the optimal exercise time for an American option given its current state (e.g., underlying asset price, time to expiration, etc.) and based on a set of available actions (e.g., exercise or hold).

To use DDPG for American ATM option pricing under GBM setting, we can follow these steps:

1. Define the environment: The first step is to define the environment, which consists of the state, action, and reward space. In the case of option pricing, the state space can be defined as the current stock price, time to maturity, and volatility, while the action space is the set of all possible exercise times. The reward space can be defined as the payoff at each exercise time, discounted by the risk-free rate.
2. Define the actor-critic network: The next step is to define the actor and critic network. The actor network is responsible for mapping states to actions, while the critic network is responsible for evaluating the quality of the action taken by the actor. Both networks can be implemented using neural networks with multiple layers.
3. Initialize the networks: The next step is to initialize the weights of the actor and critic networks. This can be done using random initialization or pre-trained weights from a similar task.
4. Set hyperparameters: The next step is to set the hyperparameters of the algorithm, such as the learning rate, discount factor, and exploration rate. The hyperparameters can be chosen through experimentation or by using common values from the literature.
5. Train the agent: The next step is to train the agent using the actor-critic algorithm. During training, the actor network learns to map states to actions that maximize the expected reward, while the critic network learns to evaluate the quality of the actor's actions. The agent learns to balance exploration and exploitation to maximize the expected reward.
6. Price the option: Once the agent is trained, we can use it to price the option. We can do this by using the actor network to determine the optimal exercise time at each point in time, and then calculating the expected payoff at each exercise time, discounted by the risk-free rate. The option price is the maximum of the expected payoffs at each exercise time.

Overall, the DDPG agent under GBM setting for American option pricing is a powerful and flexible approach that can learn an optimal exercise strategy for American options, even under complex market conditions.

### 6. **Soft Actor-Critic Agent under GBM setting**

Soft Actor-Critic (SAC) is another type of reinforcement learning algorithm that has shown great potential in solving complex decision-making problems in continuous action spaces, including the American option pricing problem.

In the context of option pricing, SAC can be used to learn an optimal policy for an agent to determine the optimal exercise strategy for an American option, given its current state and based on a set of available actions. The SAC algorithm uses a variant of the policy gradient method, which is designed to optimize the policy by taking into account both the expected future rewards and the entropy of the policy distribution.

To use SAC for American ATM option pricing under GBM setting, we can follow these steps:

1. Define the environment: The first step is to define the environment, which consists of the state, action, and reward space. In the case of option pricing, the state space can be defined as the current stock price, time to maturity, and volatility, while the action space is the set of all possible exercise times. The reward space can be defined as the payoff at each exercise time, discounted by the risk-free rate.
2. Define the actor-critic network: The next step is to define the actor and critic network. The actor network is responsible for mapping states to actions, while the critic network is responsible for evaluating the quality of the action taken by the actor. Both networks can be implemented using neural networks with multiple layers.
3. Initialize the networks: The next step is to initialize the weights of the actor and critic networks. This can be done using random initialization or pre-trained weights from a similar task.
4. Set hyperparameters: The next step is to set the hyperparameters of the algorithm, such as the learning rate, discount factor, and temperature parameter. The temperature parameter controls the trade-off between exploration and exploitation, and can be tuned to balance these two objectives.
5. Train the agent: The next step is to train the agent using the SAC algorithm. During training, the actor network learns to map states to actions that maximize the expected reward, while the critic network learns to evaluate the quality of the actor's actions. The agent learns to balance exploration and exploitation to maximize the expected reward.
6. Price the option: Once the agent is trained, we can use it to price the option. We can do this by using the actor network to determine the optimal exercise time at each point in time, and then calculating the expected payoff at each exercise time, discounted by the risk-free rate. The option price is the maximum of the expected payoffs at each exercise time.

Overall, the SAC agent under the GBM setting for American option pricing is a powerful and flexible approach that can learn an optimal exercise strategy for American options, even under complex market conditions. The combination of the policy gradient method and the GBM model can help the agent learn from a large number of training examples, leading to improved accuracy and generalization performance.

### 7. Guide Actor-Critic Agent under GBM setting

The Guide Actor-Critic Agent (GACA) is a type of reinforcement learning algorithm that can be applied to optimize the parameters of a Geometric Brownian Motion (GBM) model, which can in turn be used to price American options.  GACA_GBM.py uses the following steps to develop a GACA model:

1. Define the state space: In the GBM setting, the state space can be defined as the current stock price, the time to maturity, and the volatility of the stock price.
2. Define the action space: The action space can be defined as the parameters of the GBM model, which include the drift rate, volatility, and initial stock price.
3. Define the reward function: The reward function can be defined as the payoff of the American option at each time step.
4. Train the GACA agent: The GACA agent learns to optimize the parameters of the GBM model by maximizing the expected cumulative reward over time. The agent consists of two components: the critic network, which estimates the expected cumulative reward, and the actor network, which generates the policy for selecting actions.
5. Use the trained GBM model to price the American option: Once the GACA agent has been trained, the optimized GBM model can be used to simulate stock price paths and calculate the expected payoff of the American option.

### 8. Discussion
Comparing with SARSA/Q-learning technic we used in Assignment1, the major advantage of policy-gradient approaches is that they can handle high-dimensional action and state spaces, including actions and states that are continuous. This is because we do not have to iterate over all actions using $argmax_{\alpha \in A(S)}$ as we do in value-based approaches. For continuous problems, $argmax_{\alpha \in A(S)}$ is not possible to calculate, while for a high number of actions, the computational complexity is dependent on the number of actions.

However, a disadvantage of REINFORCE is known as sample inefficiency. Since the policy gradients algorithm takes an entire episode to do the update, it is difficult to determine which of the state-action pairs are those that effect the value G(the episode reward).

Moreover, model-free reinforcement learning is a particularly challenging case to understand and explain why a policy is making a decision. This is largely due to the model-free property: there are no action definitions that can used as these are unknown. However, policy gradients are particularly difficult because the values of states are unknown: we just have a resulting policy. With value-based approaches, knowing $V$ or $Q$ provides some insight into why actions are chosen by a policy; although explainability problems still remain.
