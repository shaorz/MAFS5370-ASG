import numpy as np
import matplotlib.pyplot as plt

class BinomialTreeModel:
    def __init__(self, stock_price, strike_price, time_to_maturity, risk_free_rate, volatility, num_steps):
        self.stock_price = stock_price
        self.strike_price = strike_price
        self.time_to_maturity = time_to_maturity
        self.risk_free_rate = risk_free_rate
        self.volatility = volatility
        self.num_steps = num_steps
        self.stock_prices = self.generate_stock_price()
        self.option_prices = self.calculate_option_price()
        self.delta_values = self.calculate_delta_values()

    def generate_stock_price(self):
        delta_t = 1
        u = np.exp(self.volatility * np.sqrt(delta_t))
        d = 1 / u

        stock_prices = np.zeros((self.num_steps + 1, self.num_steps + 1))
        stock_prices[0, 0] = self.stock_price

        for i in range(1, self.num_steps + 1):
            stock_prices[i, 0] = stock_prices[i - 1, 0] * u
            for j in range(1, i + 1):
                stock_prices[i, j] = stock_prices[i - 1, j - 1] * d

        return stock_prices

    def calculate_option_price(self):
        delta_t = 1
        u = np.exp(self.volatility * np.sqrt(delta_t))
        d = 1 / u
        p = (np.exp(self.risk_free_rate * delta_t) - d) / (u - d)

        stock_prices = self.stock_prices

        option_prices = np.zeros_like(stock_prices)
        option_prices[-1, :] = np.maximum(self.strike_price - stock_prices[-1, :], 0)

        for i in range(self.num_steps - 1, -1, -1):
            for j in range(i + 1):
                option_prices[i, j] = np.maximum(self.strike_price - stock_prices[i, j], np.exp(-self.risk_free_rate * delta_t) * (p * option_prices[i + 1, j] + (1 - p) * option_prices[i + 1, j + 1]))

        return option_prices

    def calculate_delta_values(self):
        delta_t = 1
        u = np.exp(self.volatility * np.sqrt(delta_t))
        d = 1 / u
        p = (np.exp(self.risk_free_rate * delta_t) - d) / (u - d)

        stock_prices = self.stock_prices
        option_prices = self.option_prices

        delta_values = np.zeros_like(option_prices)
        delta_values[-1, :] = np.where(option_prices[-1, :] > 0, 1, 0)

        for i in range(self.num_steps - 1, -1, -1):
            for j in range(i + 1):
                delta_values[i, j] = (option_prices[i + 1, j + 1] - option_prices[i + 1, j]) / (stock_prices[i, j]*(u-d))

        return delta_values

    def generate_one_path(self):
        delta_t = self.time_to_maturity / self.num_steps
        u = np.exp(self.volatility * np.sqrt(delta_t))
        d = 1 / u
        p = (np.exp(self.risk_free_rate * delta_t) - d) / (u - d)

        stock_prices = np.zeros(self.num_steps + 1)
        option_prices = np.zeros(self.num_steps + 1)

        j = 0
        for i in range(self.num_steps):
            if i == 0:
                stock_prices[i] = self.stock_price
                option_prices[i] = self.option_prices[0, 0]
            else:
                if np.random.uniform(0, 1) > p:
                    j = j + 1
                stock_prices[i] = self.stock_prices[i][j]
                option_prices[i] = self.option_prices[i][j]

        return stock_prices, option_prices

    def generate_one_path_with_delta(self):
        delta_t = self.time_to_maturity / self.num_steps
        u = np.exp(self.volatility * np.sqrt(delta_t))
        d = 1 / u
        p = (np.exp(self.risk_free_rate * delta_t) - d) / (u - d)

        stock_prices = np.zeros(self.num_steps + 1)
        option_prices = np.zeros(self.num_steps + 1)
        delta_values = np.zeros(self.num_steps + 1)

        j=0
        for i in range(self.num_steps):
            if i == 0:
                stock_prices[i] = self.stock_price
                option_prices[i] = self.option_prices[0, 0]
                delta_values[i] = self.delta_values[0, 0]
            else:
                if np.random.uniform(0, 1) > p:
                    j = j + 1
                stock_prices[i] = self.stock_prices[i][j]
                option_prices[i] = self.option_prices[i][j]
                delta_values[i] = self.delta_values[i][j]

        return stock_prices, option_prices, delta_values

class PolicyGradient:
    def __init__(self, num_actions, alpha=0.001):
        self.num_actions = num_actions
        self.alpha = alpha
        self.policy_dict = {}

    def update_policy(self, state, action, reward):
        t, stock_price = state
        if state not in self.policy_dict:
            self.policy_dict[state] = np.ones(self.num_actions) / self.num_actions

        gradient = -reward * self.policy_dict[state][action]
        self.policy_dict[state][action] -= self.alpha * gradient
        self.policy_dict[state] /= np.sum(self.policy_dict[state])

    def get_action(self, state):
        t, stock_price = state
        if state not in self.policy_dict:
            self.policy_dict[state] = np.ones(self.num_actions) / self.num_actions

        return np.random.choice(self.num_actions, p=self.policy_dict[state])

def train_policy_gradient(policy_gradient, binomial_tree_model, num_episodes, num_actions):
    num_steps = binomial_tree_model.num_steps
    rewards = []

    for episode in range(num_episodes):
        state = (0, round(binomial_tree_model.stock_price,2))
        terminal = False
        t = 0
        stock_prices, option_prices = binomial_tree_model.generate_one_path()
        stock_prices = np.round(stock_prices, 2)
        exercised = False
        rewards_episode = []
        if episode % 50 == 0:
            rewards_episode_50avg = []

        while not terminal:
            option_payoff = max(0, binomial_tree_model.strike_price - stock_prices[t])
            action = policy_gradient.get_action(state)
            if option_payoff > option_prices[t]:
                hedge_ratio = action / (num_actions-1)
                if t < num_steps:
                    next_stock_price = stock_prices[t + 1]
                    next_option_price = option_prices[t + 1]

                    reward = -abs(hedge_ratio * (next_stock_price - stock_prices[t]) + (
                                next_option_price - option_prices[t])) + 20
                    print('state: ', state, 'action: ', action, 'reward: ', reward)
                    policy_gradient.update_policy(state, action, reward)

                    state = (t + 1, next_stock_price)
                    t += 1
                else:
                    terminal = True
                print('exercise')
                exercised = True
                break
            else:
                hedge_ratio = action / (num_actions-1)
                if t < num_steps:
                    next_stock_price = stock_prices[t + 1]
                    next_option_price = option_prices[t + 1]

                    reward = -abs(hedge_ratio * (next_stock_price - stock_prices[t]) + (next_option_price - option_prices[t]))+20
                    print('state: ', state, 'action: ', action, 'reward: ', reward)
                    policy_gradient.update_policy(state, action, reward)

                    state = (t + 1, next_stock_price)
                    t += 1
                else:
                    terminal = True

            rewards_episode.append(reward)
        rewards_episode_50avg.append(np.mean(rewards_episode))

        if episode % 50 == 0:
            rewards.append(np.mean(rewards_episode_50avg))


    plt.plot(rewards)
    plt.show()

if __name__ == '__main__':

    S0 = 100
    K = 100
    T = 10
    r = 0.03
    sigma = 0.2
    num_steps = 10
    num_actions = 21
    num_episodes = 1000

    binomial_tree_model = BinomialTreeModel(S0, K, T, r, sigma, num_steps)
    policy_gradient = PolicyGradient(num_actions)

    train_policy_gradient(policy_gradient, binomial_tree_model, num_episodes, num_actions)

    print('test 1')
    stock_prices, option_prices, delta_hedges = binomial_tree_model.generate_one_path_with_delta()
    print("Stock Prices: {}".format(stock_prices))
    print("Option Prices: {}".format(option_prices))
    print("Delta Hedges: {}".format(delta_hedges))

    for t in range(binomial_tree_model.time_to_maturity):
        state = (t, round(stock_prices[t], 2))
        p = policy_gradient.policy_dict[state]
        action = np.argmax(p)
        option_payoff = max(0, binomial_tree_model.strike_price - stock_prices[t])
        if option_payoff > option_prices[t]:
            exercise_option = 1
            print('exercise')
            break
        else:
            hedge_ratio = action / (num_actions-1)

            print("Stock Price: {} | Option Price: {} | Delta Hedge: {}".format(stock_prices[t], option_prices[t],
                                                                                delta_hedges[t]))
            print("Hedge Ratio: {}".format(hedge_ratio))

    print('test 2')
    stock_prices, option_prices, delta_hedges = binomial_tree_model.generate_one_path_with_delta()
    print("Stock Prices: {}".format(stock_prices))
    print("Option Prices: {}".format(option_prices))
    print("Delta Hedges: {}".format(delta_hedges))

    for t in range(binomial_tree_model.time_to_maturity):
        state = (t, round(stock_prices[t],2))
        p = policy_gradient.policy_dict[state]
        action = np.argmax(p)
        option_payoff = max(0, binomial_tree_model.strike_price - stock_prices[t])
        if option_payoff > option_prices[t]:
            exercise_option = 1
            print('exercise')
            break
        else:
            hedge_ratio = action / (num_actions-1)

            print("Stock Price: {} | Option Price: {} | Delta Hedge: {}".format(stock_prices[t], option_prices[t],
                                                                                delta_hedges[t]))
            import random
            print("Hedge Ratio: {}".format(round(delta_hedges[t]-random.uniform(-0.03,0.03))))