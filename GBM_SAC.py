import numpy as np
from scipy.stats import norm
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from tqdm import trange
import random
import matplotlib.pyplot as plt


class ReplayBuffer():
    def __init__(self, max_size, input_shape):
        self.memory_size = max_size
        self.memory_cntr = 0
        self.state_memory = np.zeros((self.memory_size, *input_shape))
        self.next_state_memory = np.zeros((self.memory_size, *input_shape))
        self.action_memory = np.zeros(self.memory_size)
        self.reward_memory = np.zeros(self.memory_size)
        self.terminal_memory = np.zeros(self.memory_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.memory_cntr % self.memory_size

        self.state_memory[index] = state
        self.next_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.memory_cntr += 1

    def sample_buffer(self, batch_size):
        max_memory = min(self.memory_cntr, self.memory_size)

        batch = np.random.choice(max_memory, batch_size)

        states = self.state_memory[batch]
        states_ = self.next_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones


class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, action_dims, fc1_dims=256, fc2_dims=256, name='critic'):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.action_dims = action_dims
        self.name = name
        self.checkpoint_file = os.path.join('./', name+'_sac')

        self.fc1 = nn.Linear(self.input_dims[0]+action_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q = nn.Linear(self.fc2_dims, 1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=beta)
        self.device = torch.device("cpu")

        self.to(self.device)

    def forward(self, state, action):
        action_value = self.fc1(torch.cat([state, action], dim=1))
        action_value = F.relu(action_value)
        action_value = self.fc2(action_value)
        action_value = F.relu(action_value)

        q = self.q(action_value)

        return q

    def save_checkpoint(self):
        print('--- saving checkpoint ---')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('--- loading checkpoint ---')
        self.load_state_dict(torch.load(self.checkpoint_file))


class ValueNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims=256, fc2_dims=256, name='value'):
        super(ValueNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name
        self.checkpoint_file = os.path.join('./', name+'_sac')

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, fc2_dims)
        self.v = nn.Linear(self.fc2_dims, 1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=beta)
        self.device = torch.device("cpu")

        self.to(self.device)

    def forward(self, state):
        state_value = self.fc1(state)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = F.relu(state_value)

        v = self.v(state_value)

        return v

    def save_checkpoint(self):
        print('--- saving checkpoint ---')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('--- loading checkpoint ---')
        self.load_state_dict(torch.load(self.checkpoint_file))


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, max_action, fc1_dims=256, fc2_dims=256, action_dims=1, name='actor'):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.action_dims = action_dims
        self.name = name
        self.checkpoint_file = os.path.join('./', name+'_sac')
        self.max_action = max_action
        self.reparam_noise = 1e-6

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.action_dims)
        self.sigma = nn.Linear(self.fc2_dims, self.action_dims)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device("cpu")

        self.to(self.device)

    def forward(self, state):
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)

        mu = self.mu(prob)
        sigma = self.sigma(prob)

        sigma = torch.clamp(sigma, min=self.reparam_noise, max=1)

        return mu, sigma

    def sample_normal(self, state, reparameterize=True):
        mu, sigma = self.forward(state)
        probabilities = torch.distributions.Normal(mu, sigma)

        if reparameterize:
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()

        action = torch.tanh(actions)*torch.tensor(self.max_action).to(self.device)
        log_probs = probabilities.log_prob(actions)
        log_probs -= torch.log(1-action.pow(2)+self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim=True)

        return action, log_probs

    def save_checkpoint(self):
        print('--- saving checkpoint ---')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('--- loading checkpoint ---')
        self.load_state_dict(torch.load(self.checkpoint_file))


class Agent():
    def __init__(self, alpha=0.0003, beta=0.0003, input_dims=[2], gamma=0.99, action_dims=1, max_size=1000000, tau=0.005,
                 layer1_size=256, layer2_size=256, batch_size=256, reward_scale=2):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims)
        self.batch_size = batch_size
        self.action_dims = action_dims

        self.actor = ActorNetwork(alpha, input_dims, action_dims=action_dims, name='actor', max_action=1)
        self.critic_1 = CriticNetwork(beta, input_dims, action_dims=action_dims, name='critic_1')
        self.critic_2 = CriticNetwork(beta, input_dims, action_dims=action_dims, name='critic_2')
        self.value = ValueNetwork(beta, input_dims, name='value')
        self.target_value = ValueNetwork(beta, input_dims, name='target_value')

        self.scale = reward_scale
        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        state = torch.Tensor([observation]).to(self.actor.device)
        actions, _ = self.actor.sample_normal(state, reparameterize=False)

        return actions.cpu().detach().numpy()[0]

    def choose_final_action(self, observation):
        self.actor.eval()
        state = torch.Tensor([observation]).to(self.actor.device)
        mu, sigma = self.actor.forward(state)
        action = torch.tanh(mu)*torch.tensor(self.actor.max_action)
        action = action.to(self.actor.device).cpu().detach().numpy()[0]

        return max(action, 0)

    def remember(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state, done)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()

        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        for name in value_state_dict:
            value_state_dict[name] = tau * value_state_dict[name].clone() + (1 - tau) * target_value_state_dict[name].clone()

        self.target_value.load_state_dict(value_state_dict)

    def save_models(self):
        print('--- saving models ---')
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()

    def load_models(self):
        print('--- loading models ---')
        self.actor.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()

    def learn(self):
        if self.memory.memory_cntr < self.batch_size:
            return

        state, action, reward, next_state, done = self.memory.sample_buffer(self.batch_size)

        reward = torch.tensor(reward, dtype=torch.float).to(self.actor.device)
        done = torch.tensor(done).to(self.actor.device)
        state_ = torch.tensor(next_state, dtype=torch.float).to(self.actor.device)
        state = torch.tensor(state, dtype=torch.float).to(self.actor.device)
        action = [action]
        action = torch.tensor(action, dtype=torch.float).to(self.actor.device)
        action = action.T
        value = self.value(state).view(-1)
        value_ = self.target_value(state_).view(-1)
        value_[done] = 0.0

        actions, log_probs = self.actor.sample_normal(state, reparameterize=False)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = torch.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        self.value.optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * F.mse_loss(value, value_target)
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = torch.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        actor_loss = log_probs - critic_value
        actor_loss = torch.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        q_hat = self.scale * reward + self.gamma * value_
        q1_old_policy = self.critic_1.forward(state, action).view(-1)
        q2_old_policy = self.critic_2.forward(state, action).view(-1)
        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)

        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.update_network_parameters()

class GeometricBrownianMotionStock(object):
    def __init__(self, initial_price, strike_price, risk_free_rate, volatility, num_steps):
        self.initial_price = initial_price
        self.current_price = initial_price
        self.strike_price = strike_price
        self.risk_free_rate = risk_free_rate
        self.volatility = volatility
        self.num_steps = num_steps
        self.days_per_year = 252
        self.time_step = 1 / self.days_per_year
        self.total_time = num_steps / self.days_per_year

    def compute_put_option_price(self, price, strike, rate, vol, time_to_maturity):
        if time_to_maturity == 0:
            return max(strike - price, 0)
        else:
            d1 = (np.log(price / strike) + (rate + vol ** 2 / 2) * time_to_maturity) / (vol * np.sqrt(time_to_maturity))
            d2 = d1 - vol * np.sqrt(time_to_maturity)
        return float(-price * norm.cdf(-d1) + strike * np.exp(-rate * time_to_maturity) * norm.cdf(-d2))

    def reset(self):
        state = [self.total_time, self.initial_price]
        return state

    def compute_put_delta(self, price, strike, rate, vol, time_to_maturity):
        if time_to_maturity == 0:
            return float(np.where(strike >= price, 1, 0))
        else:
            d1 = (np.log(price / strike) + (rate + vol ** 2 / 2) * time_to_maturity) / (vol * np.sqrt(time_to_maturity))
        return float(-norm.cdf(-d1))

    def step(self, action, exercised):
        current_option = self.compute_put_option_price(self.current_price, self.strike_price, self.risk_free_rate,
                                                       self.volatility, self.total_time)
        current_stock_price = self.current_price
        self.total_time -= self.time_step
        self.current_price = self.current_price * np.exp((self.risk_free_rate - self.volatility ** 2 / 2) * self.time_step + self.volatility * np.random.normal() * np.sqrt(self.time_step))

        if exercised:
            reward = action * (self.current_price - current_stock_price)
        else:
            current_option_payoff = np.maximum(self.strike_price - current_stock_price, 0)
            if current_option_payoff > current_option:
                exercised = True
                reward = action * (self.current_price - current_stock_price)
            else:
                next_option = self.compute_put_option_price(self.current_price, self.strike_price, self.risk_free_rate,
                                                            self.volatility, self.total_time)
                reward = next_option - current_option + action * (self.current_price - current_stock_price)

        reward = -abs(reward)
        next_state = [self.total_time, self.initial_price]

        if self.total_time <= 0 or np.isclose(self.total_time, 0):
            completed = True
        else:
            completed = False

        return next_state, reward, completed, exercised

    def generate_one_path(self):
        stock_price_path = []
        option_price_path = []
        delta_path = []
        for i in range(self.num_steps):
            current_option = self.compute_put_option_price(self.current_price, self.strike_price, self.risk_free_rate,
                                                       self.volatility, self.total_time)
            current_stock_price = self.initial_price
            self.total_time -= self.time_step
            self.current_price = self.current_price * np.exp((self.risk_free_rate - self.volatility ** 2 / 2) * self.time_step + self.volatility * np.random.normal() * np.sqrt(self.time_step))
            stock_price_path.append(self.current_price)
            option_price_path.append(current_option)
            delta_path.append(self.compute_put_delta(self.current_price, self.strike_price, self.risk_free_rate, self.volatility, self.total_time))

        return stock_price_path, option_price_path, delta_path

if __name__ == '__main__':
    episodes = 20000

    agent = Agent(alpha=0.0003, beta=0.0003, tau=0.001)

    rewards = []
    for i in trange(episodes):
        env = GeometricBrownianMotionStock(initial_price=100, strike_price=100, risk_free_rate=0.03, volatility=0.3, num_steps=10)
        completed = False
        exercised = False
        score = 0
        state = env.reset()
        while not completed:
            action = agent.choose_action(state)
            next_state, reward, completed, exercised = env.step(action, exercised)
            agent.remember(state, action, reward, next_state, int(completed))
            agent.learn()
            score += reward
            state = next_state

        reward_episode = []
        num_steps = 10
        for i in range(10):
            env = GeometricBrownianMotionStock(initial_price=100, strike_price=100, risk_free_rate=0.03, volatility=0.3, num_steps=10)
            completed = False
            exercised = False
            score = 0
            state = env.reset()
            while not completed:
                action = agent.choose_final_action(state)
                next_state, reward, completed, exercised = env.step(action, exercised)
                score += reward
                state = next_state

            reward_episode.append(score)

        rewards.append(np.mean(reward_episode))

    plt.plot(rewards)
    plt.show()

    print('test 1')
    random.seed(0)
    env = GeometricBrownianMotionStock(initial_price=100, strike_price=100, risk_free_rate=0.03, volatility=0.3, num_steps=10)
    a,b, delta_hedging = env.generate_one_path()
    print(delta_hedging)
    actions = []
    num_steps = 10
    env = GeometricBrownianMotionStock(initial_price=100, strike_price=100, risk_free_rate=0.03, volatility=0.3, num_steps=10)
    for i in range(10):
        completed = False
        exercised = False
        score = 0
        state = env.reset()
        while not completed:
            action = agent.choose_final_action(state)
            actions.append(action)
            next_state, reward, completed, exercised = env.step(action, exercised)
            score += reward
            state = next_state
    print(actions)