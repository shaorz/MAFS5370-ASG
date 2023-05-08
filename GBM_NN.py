import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import scipy

class GBMModel:
    def __init__(self, stock_price, strike_price, time_to_maturity, risk_free_rate, volatility, num_paths, num_steps):
        self.stock_price = stock_price
        self.strike_price = strike_price
        self.time_to_maturity = time_to_maturity
        self.risk_free_rate = risk_free_rate
        self.volatility = volatility
        self.num_paths = num_paths
        self.num_steps = num_steps

    # def generate_stock_price(self):
    #     dt = self.time_to_maturity / self.num_steps
    #     stock_prices = np.zeros((self.num_paths, self.num_steps + 1))
    #     stock_prices[:, 0] = self.stock_price
    #
    #     for t in range(1, self.num_steps + 1):
    #         z = np.random.standard_normal(self.num_paths)
    #         stock_prices[:, t] = stock_prices[:, t - 1] * np.exp((self.risk_free_rate - 0.5 * self.volatility ** 2) * dt + self.volatility * np.sqrt(dt) * z)
    #
    #     return stock_prices

    # def calculate_option_price(self):
    #     dt = self.time_to_maturity / self.num_steps
    #     stock_prices = self.stock_prices
    #     cashflows = np.maximum(self.strike_price - stock_prices[:, -1], 0)
    #
    #     for t in range(self.num_steps - 1, 0, -1):
    #         in_money = stock_prices[:, t] < self.strike_price
    #         X = stock_prices[in_money, t]
    #         Y = cashflows[in_money] * np.exp(-self.risk_free_rate * dt)
    #
    #         A = np.vstack([np.ones_like(X), X, X ** 2]).T
    #         beta = np.linalg.lstsq(A, Y, rcond=None)[0]
    #         continuation_value = A @ beta
    #         exercise_value = self.strike_price - X
    #
    #         exercise_now = exercise_value > continuation_value
    #         cashflows[in_money] = np.where(exercise_now, exercise_value, cashflows[in_money] * np.exp(-self.risk_free_rate * dt))
    #
    #     option_price = np.mean(cashflows * np.exp(-self.risk_free_rate * dt))
    #     return option_price

    def generate_stock_price(self):
        delta_t = self.time_to_maturity / self.num_steps
        stock_prices = np.zeros(self.num_steps + 1)
        stock_prices[0] = self.stock_price

        for i in range(1, self.num_steps + 1):
            z = np.random.normal(0, 1)
            stock_prices[i] = stock_prices[i - 1] * np.exp((self.risk_free_rate - (self.volatility ** 2) / 2) * delta_t + self.volatility * np.sqrt(delta_t) * z)

        return stock_prices

    def calculate_option_price(self, stock_prices):
        def black_scholes(stock_price, strike_price, time_to_maturity, risk_free_rate, volatility):
            d1 = (np.log(stock_price / strike_price) + (risk_free_rate + (volatility ** 2) / 2) * time_to_maturity) / (volatility * np.sqrt(time_to_maturity))
            d2 = d1 - volatility * np.sqrt(time_to_maturity)

            put_price = strike_price * np.exp(-risk_free_rate * time_to_maturity) * scipy.stats.norm.cdf(-d2) - stock_price * scipy.stats.norm.cdf(-d1)

            return put_price

        stock_prices = stock_prices
        option_prices = np.zeros(self.num_steps + 1)

        for i in range(self.num_steps + 1):
            time_remaining = (self.num_steps - i) * (self.time_to_maturity / self.num_steps)
            put_price = black_scholes(stock_prices[i], self.strike_price, time_remaining, self.risk_free_rate, self.volatility)
            option_prices[i] = put_price

        return option_prices

class DeepPolicyGradient(nn.Module):
    def __init__(self, num_actions, alpha=0.001, epsilon=0.9):
        super(DeepPolicyGradient, self).__init__()
        self.num_actions = num_actions
        self.alpha = alpha
        self.epsilon = epsilon
        self.model = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
            nn.Softmax(dim=-1)
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)

    def forward(self, state):
        return self.model(state)

    def update_policy(self, state, action, reward):
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.model(state)
        log_prob = probs[0, action]
        loss = -reward * log_prob
        print(state, action, reward, loss)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.model(state).detach().numpy()[0]
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.randint(self.num_actions)
        else:
            return np.random.choice(self.num_actions, p=probs)

# Train the deep policy gradient model with the GBMModel class
def train_deep_policy_gradient(policy_gradient, gbm_model, num_episodes, gamma=0.99):
    num_steps = gbm_model.num_steps
    rewards = []

    for episode in range(num_episodes):
        state = (0, round(gbm_model.stock_price, 2))
        terminal = False
        t = 0
        stock_prices = gbm_model.generate_stock_price()
        option_prices = gbm_model.calculate_option_price(stock_prices)
        stock_prices = np.round(stock_prices, 2)

        while not terminal:
            action = policy_gradient.get_action(state)
            if action == 11:
                reward = 0
                policy_gradient.update_policy(state, action, reward)
                print('exercise')
                break
            else:
                hedge_ratio = action / 10
                if t < num_steps:
                    next_stock_price = stock_prices[t + 1]
                    next_option_price = option_prices[t + 1]

                    reward = -abs(hedge_ratio * (next_stock_price - stock_prices[t]) + (
                                next_option_price - option_prices[t])) + 20
                    rewards.append(reward)
                    print('state: ', state, 'action: ', action, 'reward: ', reward)
                    policy_gradient.update_policy(state, action, reward)

                    state = (t + 1, next_stock_price)
                    t += 1
                else:
                    terminal = True

    plt.plot(rewards)
    plt.show()

# Initialize the parameters
stock_price = 100
strike_price = 100
time_to_maturity = 1
risk_free_rate = 0.05
volatility = 0.2
num_paths = 10000
num_steps = 50
num_actions = 12
num_episodes = 1000

# Create the GBM model and policy gradient model
gbm_model = GBMModel(stock_price, strike_price, time_to_maturity, risk_free_rate, volatility, num_paths, num_steps)
policy_gradient = DeepPolicyGradient(num_actions)

# Train the policy gradient model
train_deep_policy_gradient(policy_gradient, gbm_model, num_episodes)

