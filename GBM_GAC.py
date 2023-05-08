import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal
from scipy import optimize
from scipy.stats import norm

# Architecture for policy
hidden_layer_sep_pol = 2
hidden_units_list_pol = [400, 300]

# Architecture for q
hidden_layer_sep_q = 1  # Number of hidden layer for separated s and a branches
hidden_units_list_sep_q = [400]  # Number of hidden units in each layer for separate the branches
hidden_layer_merge_q = 1  # Number of hidden layer for the concatenated layer from the branches
hidden_units_list_merge_q = [300]  # Number of hidden units in the concatenated

initializer = nn.init.xavier_uniform_  # Weight initializer
final_initializer = nn.init.uniform_  # Weight initializer for the final layer
hidden_activation = nn.ReLU

class GAC_learner:
    def __init__(self, ds, da, epsilon=0.0001, entropy_gamma=0.99, n_taylor=1, min_cov=0.01, \
                 gamma=0.99, tau=0.001, lr_q=0.001, lr_policy=0.0001, \
                 reg_q=0, reg_policy=0, action_bnds=[-1, 1], log_level=0):
        """ Initialize the learner object """
        self.ds = ds
        self.da = da
        self.gamma = gamma
        self.epsilon = epsilon
        self.entropy_gamma = entropy_gamma
        self.tau = tau
        self.optimizer_Q = optim.Adam(lr=lr_q)
        self.optimizer_policy = optim.Adam(lr=lr_policy)
        self.reg_q = nn.MSELoss()
        self.reg_policy = nn.MSELoss()
        self.entropy = 0
        self.cov = np.identity(self.da)
        self.n_taylor = n_taylor
        self.log_level = log_level
        self.action_bnds = action_bnds

        if action_bnds[0] == -action_bnds[1]:
            self.a_scale = action_bnds[1]
        else:
            print("Action's upper-bound and lower-bound must be symmetric.")
            return

        if min_cov <= 0:
            min_cov = 0.01

        # Compute a minimum policy's entropy using min_cov
        self.entropy_0 = 0.5 * np.log(min_cov) * self.da + 0.5 * self.da * (
                    1 + np.log(2 * np.pi))  # The minimum entropy
        print("Minimum entropy %f." % (self.entropy_0))

        # Compute the current policy's entropy using current cov
        (sign, logdet) = np.linalg.slogdet(self.cov)
        entropy_init = 0.5 * sign * logdet + 0.5 * self.da * (1 + np.log(2 * np.pi))
        self.beta = self.entropy_gamma * (entropy_init - self.entropy_0) + self.entropy_0  # Initial entropy bound
        print("Initial entropy %f from an identity matrix." % entropy_init)

        # Create the actor network, the critic network, and the target critic network.
        self.deep_mean_model = self.create_policy_network()
        self.deep_q_model, self.state_critic, self.action_critic, self.action_grads \
            = self.create_q_network()
        self.target_deep_q_model, _, _, _ = self.create_q_network()

        # Copy the initial weights of the critic network to the target critic network
        self.update_target_network()

    def create_policy_network(self):
        """ Create the actor (policy) network """
        layers = [nn.Linear(self.ds, hidden_units_list_pol[0])]
        initializer(layers[-1].weight)
        layers.append(hidden_activation())

        for i in range(1, hidden_layer_sep_pol):
            layers.append(nn.Linear(hidden_units_list_pol[i - 1], hidden_units_list_pol[i]))
            initializer(layers[-1].weight)
            layers.append(hidden_activation())

        layers.append(nn.Linear(hidden_units_list_pol[-1], self.da))
        final_initializer(layers[-1].weight, a=-0.003, b=0.003)

        return nn.Sequential(*layers)

    def create_q_network(self):
        """ Create the critic (Q) network """
        state_critic = [nn.Linear(self.ds, hidden_units_list_sep_q[0])]
        initializer(state_critic[-1].weight)
        state_critic.append(hidden_activation())

        for i in range(1, hidden_layer_sep_q):
            state_critic.append(nn.Linear(hidden_units_list_sep_q[i - 1], hidden_units_list_sep_q[i]))
            initializer(state_critic[-1].weight)
            state_critic.append(hidden_activation())

        action_critic = [nn.Linear(self.da, hidden_units_list_sep_q[0])]
        initializer(action_critic[-1].weight)
        action_critic.append(hidden_activation())

        for i in range(1, hidden_layer_sep_q):
            action_critic.append(nn.Linear(hidden_units_list_sep_q[i - 1], hidden_units_list_sep_q[i]))
            initializer(action_critic[-1].weight)
            action_critic.append(hidden_activation())

        # Merge the state and action branches
        merge_layer = [nn.Linear(hidden_units_list_sep_q[-1] * 2, hidden_units_list_merge_q[0])]
        initializer(merge_layer[-1].weight)
        merge_layer.append(hidden_activation())

        for i in range(1, hidden_layer_merge_q):
            merge_layer.append(nn.Linear(hidden_units_list_merge_q[i - 1], hidden_units_list_merge_q[i]))
            initializer(merge_layer[-1].weight)
            merge_layer.append(hidden_activation())

        merge_layer.append(nn.Linear(hidden_units_list_merge_q[-1], 1))
        final_initializer(merge_layer[-1].weight, a=-0.003, b=0.003)

        deep_q_model = nn.Sequential(*merge_layer)

        return deep_q_model, state_critic, action_critic, merge_layer

    def update_target_network(self):
        """ Update the target critic network with a soft update """
        for target_param, param in zip(self.target_deep_q_model.parameters(), self.deep_q_model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def get_action(self, state):
        """ Get the action from the policy network """
        with torch.no_grad():
            mean = self.deep_mean_model(torch.tensor(state, dtype=torch.float32))
            action = torch.clamp(mean, min=self.action_bnds[0], max=self.action_bnds[1]).numpy()

        return action


class GeometricBrownianMotionStock:
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
    print('NOT YET FINISHED')