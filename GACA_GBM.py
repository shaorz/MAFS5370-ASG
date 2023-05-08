import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Set the option parameters
S0 = 100   # Stock price
K = 100    # Strike price
T = 10     # Time to maturity (in years)
sigma = 0.2   # Volatility
r = 0.05   # Risk-free rate

# Define the Black-Scholes model for European put option pricing
def bs_put(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    Nd1 = tfp.distributions.Normal(0, 1).cdf(-d1)
    Nd2 = tfp.distributions.Normal(0, 1).cdf(-d2)
    return K*tf.math.exp(-r*T)*Nd2 - S*Nd1

# Define the state and action spaces
state_dim = 1  # Stock price
action_dim = 1  # Exercise decision (0 for hold, 1 for exercise)

# Define the actor and critic networks
def create_actor_network(state_dim, action_dim):
    state_input = Input(shape=(state_dim,))
    x = Dense(32, activation='relu')(state_input)
    x = Dense(32, activation='relu')(x)
    action_output = Dense(action_dim, activation='sigmoid')(x)
    model = Model(inputs=state_input, outputs=action_output)
    return model

def create_critic_network(state_dim):
    state_input = Input(shape=(state_dim,))
    x = Dense(32, activation='relu')(state_input)
    x = Dense(32, activation='relu')(x)
    value_output = Dense(1)(x)
    model = Model(inputs=state_input, outputs=value_output)
    return model

# Define the GAC agent
class GACAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actor_network = create_actor_network(state_dim, action_dim)
        self.critic_network = create_critic_network(state_dim)
        self.actor_optimizer = Adam(lr=0.001)
        self.critic_optimizer = Adam(lr=0.001)
    
    def act(self, state):
        action_prob = self.actor_network.predict(np.array([state]))[0]
        action_dist = tfp.distributions.Bernoulli(probs=action_prob)
        action = action_dist.sample()
        return action.numpy()[0]
    
    def learn(self, state, action, reward, next_state, done):
        target_value = reward
        if not done:
            target_value += 0.99 * self.critic_network.predict(np.array([next_state]))[0][0]
        
        advantage = target_value - self.critic_network.predict(np.array([state]))[0][0]
        
        with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape:
            actor_prob = self.actor_network(np.array([state]))
            action_dist = tfp.distributions.Bernoulli(probs=actor_prob)
            log_prob = action_dist.log_prob(action)
            
            actor_loss = -log_prob * advantage
            critic_loss = advantage**2
            
            actor_grads = actor_tape.gradient(actor_loss, self.actor

                                              
def main():
    # Set the option parameters
    S0 = 100   # Stock price
    K = 100    # Strike price
    T = 10     # Time to maturity (in years)
    sigma = 0.2   # Volatility
    r = 0.05   # Risk-free rate
    
    # Create the GAC agent
    agent = GACAgent(state_dim, action_dim)
    
    # Train the agent
    episode_reward = 0
    state = S0
    for t in range(1000):
        action = agent.act(state)
        reward = -bs_put(state, K, T, r, sigma)
        
        next_state = state - 1
        if action == 1 or t == 999:
            done = True
        else:
            done = False
        
        agent.learn(state, action, reward, next_state, done)
        
        state = next_state
        episode_reward += reward
        
        if done:
            print('Episode: {}, Reward: {:.2f}'.format(t+1, episode_reward))
            break
    
    # Price the American ATM put option
    option_price = bs_put(S0, K, T, r, sigma)
    print('Option price (European): {:.2f}'.format(option_price))
    
    state = S0
    while True:
        action = agent.act(state)
        if action == 1:
            option_price = bs_put(state, K, T, r, sigma)
            print('Option price (American): {:.2f}'.format(option_price))
            break
        else:
            state -= 1
