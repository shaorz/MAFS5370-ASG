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


class ReplayBuffer(object):
    def __init__(self, max_size, input_dim):
        self.memory_size = max_size
        self.memory_cntr = 0
        self.state_memory = np.zeros((self.memory_size, *input_dim))
        self.next_state_memory = np.zeros((self.memory_size, *input_dim))
        self.action_memory = np.zeros(self.memory_size)
        self.reward_memory = np.zeros(self.memory_size)
        self.terminal_memory = np.zeros(self.memory_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done):
        index = self.memory_cntr % self.memory_size
        self.state_memory[index] = state
        self.next_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - done
        self.memory_cntr += 1

    def sample_buffer(self, batch_size):
        max_memory = min(self.memory_cntr, self.memory_size)

        batch = np.random.choice(max_memory, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        terminal = self.terminal_memory[batch]
        next_states = self.next_state_memory[batch]

        return states, actions, rewards, next_states, terminal


class OUActionNoise(object):
    def __init__(self, mu, sigma=0.1, theta=0.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)


class CriticNetwork(nn.Module):
    def __init__(self, alpha, input_dim, fc1_dim, fc2_dim, action_dim, name):
        super(CriticNetwork, self).__init__()
        self.input_dim = input_dim
        self.fc1_dim = fc1_dim
        self.fc2_dim = fc2_dim
        self.action_dim = action_dim
        self.save_checkpoint_file = os.path.join('./', name+'_ddpg')
        self.fc1 = nn.Linear(*self.input_dim, self.fc1_dim)
        f1 = 1. / np.sqrt(self.fc1.weight.data.size()[0])
        nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        self.bn1 = nn.LayerNorm(self.fc1_dim)

        self.fc2 = nn.Linear(self.fc1_dim, self.fc2_dim)
        f2 = 1. / np.sqrt(self.fc2.weight.data.size()[0])
        nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        self.bn2 = nn.LayerNorm(self.fc2_dim)
        self.action_value = nn.Linear(self.action_dim, self.fc2_dim)
        f3 = 0.003
        self.q = nn.Linear(self.fc2_dim, 1)
        nn.init.uniform_(self.q.weight.data, -f3, f3)
        nn.init.uniform_(self.q.bias.data, -f3, f3)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")

        self.to(self.device)

    def forward(self, state, action):
        x = self.fc1(state)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        action_value = F.relu(self.action_value(action))
        state_action_value = F.relu(torch.add(x, action_value))
        state_action_value = self.q(state_action_value)

        return state_action_value

    def save_checkpoint(self):
        print('--- saving checkpoint ---')
        torch.save(self.state_dict(), self.save_checkpoint_file)

    def load_checkpoint(self):
        print('--- loading checkpoint ---')
        self.load_state_dict(torch.load(self.save_checkpoint_file))


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dim, fc1_dim, fc2_dim, action_dim, name):
        super(ActorNetwork, self).__init__()
        self.input_dim = input_dim
        self.fc1_dim = fc1_dim
        self.fc2_dim = fc2_dim
        self.action_dim = action_dim
        self.fc1 = nn.Linear(*self.input_dim, self.fc1_dim)
        f1 = 1. / np.sqrt(self.fc1.weight.data.size()[0])
        nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        self.bn1 = nn.LayerNorm(self.fc1_dim)

        self.fc2 = nn.Linear(self.fc1_dim, self.fc2_dim)
        f2 = 1. / np.sqrt(self.fc2.weight.data.size()[0])
        nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        self.bn2 = nn.LayerNorm(self.fc2_dim)

        f3 = 0.003
        self.mu = nn.Linear(self.fc2_dim, self.action_dim)
        nn.init.uniform_(self.mu.weight.data, -f3, f3)
        nn.init.uniform_(self.mu.bias.data, -f3, f3)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        print(self.device)

        self.to(self.device)
        self.save_checkpoint_file = os.path.join('./', name+'_ddpg')

    def forward(self, state):
        x = self.fc1(state)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.tanh(self.mu(x))

        return torch.abs(x)

    def save_checkpoint(self):
        print('--- saving checkpoint ---')
        torch.save(self.state_dict(), self.save_checkpoint_file)

    def load_checkpoint(self):
        print('--- loading checkpoint ---')
        self.load_state_dict(torch.load(self.save_checkpoint_file))


class Agent(object):
    def __init__(self, alpha, beta, input_dim, tau, gamma=0.99,
                 action_dim=1, max_size=1000000, layer1_size=400,
                 layer2_size=300, batch_size=64):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dim)
        self.batch_size = batch_size

        self.actor = ActorNetwork(alpha, input_dim, layer1_size, layer2_size, action_dim=action_dim, name='Actor')
        self.critic = CriticNetwork(beta, input_dim, layer1_size, layer2_size, action_dim=action_dim, name='Critic')

        self.target_actor = ActorNetwork(alpha, input_dim, layer1_size, layer2_size, action_dim=action_dim, name='TargetActor')
        self.target_critic = CriticNetwork(beta, input_dim, layer1_size, layer2_size, action_dim=action_dim, name='TargetCritic')

        self.noise = OUActionNoise(mu=np.zeros(action_dim))

        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        self.actor.eval()
        observation = torch.tensor(observation, dtype=torch.float).to(self.actor.device)
        mu = self.actor.forward(observation).to(self.actor.device)
        mu_prime = mu + torch.tensor(self.noise(), dtype=torch.float).to(self.actor.device)
        self.actor.train()
        return max(mu_prime.cpu().detach().numpy(), 0)

    def choose_final_action(self, observation):
        self.actor.eval()
        observation = torch.tensor(observation, dtype=torch.float).to(self.actor.device)
        mu = self.actor(observation).to(self.actor.device)
        self.actor.train()
        return max(mu.cpu().detach().numpy(), 0)

    def remember(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state, done)

    def learn(self):
        if self.memory.memory_cntr < self.batch_size:
            return
        state, action, reward, next_state, done = self.memory.sample_buffer(self.batch_size)

        reward = torch.tensor(reward, dtype=torch.float).to(self.critic.device)
        done = torch.tensor(done).to(self.critic.device)
        next_state = torch.tensor(next_state, dtype=torch.float).to(self.critic.device)
        action = [action]
        action = torch.tensor(action, dtype=torch.float).to(self.critic.device)
        action = action.T
        state = torch.tensor(state, dtype=torch.float).to(self.critic.device)

        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.eval()
        target_actions = self.target_actor.forward(next_state)
        critic_value_ = self.target_critic.forward(next_state, target_actions)
        critic_value = self.critic.forward(state, action)

        target = []
        for j in range(self.batch_size):
            target.append(reward[j] + self.gamma * critic_value_[j] * done[j])
        target = torch.tensor(target).to(self.critic.device)
        target = target.view(self.batch_size, 1)

        self.critic.train()
        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()

        self.critic.eval()
        self.actor.optimizer.zero_grad()
        mu = self.actor.forward(state)
        self.actor.train()
        actor_loss = -self.critic.forward(state, mu)
        actor_loss = torch.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_dict = dict(target_critic_params)
        target_actor_dict = dict(target_actor_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau * critic_state_dict[name].clone() + (1 - tau) * target_critic_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)

        for name in actor_state_dict:
            actor_state_dict[name] = tau * actor_state_dict[name].clone() + (1 - tau) * target_actor_dict[name].clone()
        self.target_actor.load_state_dict(actor_state_dict)

    def save_models(self):
        print('--- saving the best model ---')
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        print('--- loading the best model ---')
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()

    def check_actor_params(self):
        current_actor_params = self.actor.named_parameters()
        current_actor_dict = dict(current_actor_params)
        original_actor_dict = dict(self.original_actor.named_parameters())
        original_critic_dict = dict(self.original_critic.named_parameters())
        current_critic_params = self.critic.named_parameters()
        current_critic_dict = dict(current_critic_params)
        print('Checking Actor parameters')

        for param in current_actor_dict:
            print(param, torch.equal(original_actor_dict[param], current_actor_dict[param]))
        print('Checking critic parameters')
        for param in current_critic_dict:
            print(param, torch.equal(original_critic_dict[param], current_critic_dict[param]))
        input()


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

    agent = Agent(alpha=0.00001, beta=0.00001, input_dim=[2], tau=0.001)

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

