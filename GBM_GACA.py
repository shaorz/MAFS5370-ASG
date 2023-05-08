import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Set the option parameters
S0 = 100  # Stock price
K = 100  # Strike price
T = 10  # Time to maturity (in years)
sigma = 0.2  # Volatility
r = 0.05  # Risk-free rate


class Actor ( nn.Module ):
	def __init__ ( self , state_dim , action_dim ):
		super ( Actor , self ).__init__ ()
		self.fc1 = nn.Linear ( state_dim , 128 )
		self.fc2 = nn.Linear ( 128 , 64 )
		self.fc3 = nn.Linear ( 64 , action_dim )

	def forward ( self , state ):
		x = F.relu ( self.fc1 ( state ) )
		x = F.relu ( self.fc2 ( x ) )
		x = torch.tanh ( self.fc3 ( x ) )
		return x


class Critic ( nn.Module ):
	def __init__ ( self , state_dim ):
		super ( Critic , self ).__init__ ()
		self.fc1 = nn.Linear ( state_dim , 128 )
		self.fc2 = nn.Linear ( 128 , 64 )
		self.fc3 = nn.Linear ( 64 , 1 )

	def forward ( self , state ):
		x = F.relu ( self.fc1 ( state ) )
		x = F.relu ( self.fc2 ( x ) )
		x = self.fc3 ( x )
		return x


class GACA_GBM:
	def __init__ ( self , state_dim , action_dim , gamma = 0.99 , tau = 0.005 , lr = 0.001 ):
		self.actor = Actor ( state_dim , action_dim )
		self.actor_target = Actor ( state_dim , action_dim )
		self.critic = Critic ( state_dim )
		self.critic_target = Critic ( state_dim )
		self.actor_optimizer = optim.Adam ( self.actor.parameters () , lr = lr )
		self.critic_optimizer = optim.Adam ( self.critic.parameters () , lr = lr )
		self.gamma = gamma
		self.tau = tau
		self.soft_update ( self.actor , self.actor_target , 1.0 )
		self.soft_update ( self.critic , self.critic_target , 1.0 )

	def act ( self , state , exploration_noise = 0.1 ):
		state = torch.FloatTensor ( state )
		action = self.actor ( state ).detach ().numpy ()
		action += exploration_noise * np.random.randn ( self.actor.out_features )
		action = np.clip ( action , -1.0 , 1.0 )
		return action

	def q_value ( self , state , action ):
		state = torch.FloatTensor ( state )
		action = torch.FloatTensor ( action )
		q_value = self.critic ( torch.cat ( [ state , action ] , 1 ) )
		return q_value

	def q_target ( self , next_state , next_action ):
		next_state = torch.FloatTensor ( next_state )
		next_action = torch.FloatTensor ( next_action )
		q_target = self.critic_target ( torch.cat ( [ next_state , next_action ] , 1 ) )
		return q_target

	def update_target_networks ( self ):
		for target_param , param in zip ( self.actor_target.parameters () , self.actor.parameters () ):
			target_param.data.copy_ ( self.tau * param.data + (1 - self.tau) * target_param.data )

		for target_param , param in zip ( self.critic_target.parameters () , self.critic.parameters () ):
			target_param.data.copy_ ( self.tau * param.data + (1 - self.tau) * target_param.data )

	def soft_update ( self , net , target_net , tau ):
		for target_param , param in zip ( target_net.parameters () , net.parameters () ):
			target_param.data.copy_ ( tau * param.data + (1 - tau) * target_param.data )

	def learn ( self , replay_buffer , iterations , batch_size = 64 , gamma = 0.99 , soft_tau = 1e-2 ):
		for it in range ( iterations ):
			# Sample replay buffer
			state , action , reward , next_state , done = replay_buffer.sample ( batch_size )
			state = torch.FloatTensor ( state ).to ( self.device )
			next_state = torch.FloatTensor ( next_state ).to ( self.device )
			action = torch.FloatTensor ( action ).to ( self.device )
			reward = torch.FloatTensor ( reward ).to ( self.device )
			done = torch.FloatTensor ( done ).to ( self.device )

			# Compute the target Q value
			target_Q = self.critic_target ( next_state , self.actor_target ( next_state ) )
			target_Q = reward + ((1 - done) * gamma * target_Q).detach ()

			# Get current Q estimate
			current_Q = self.critic ( state , action )

			# Compute critic loss
			critic_loss = F.mse_loss ( current_Q , target_Q )

			# Optimize the critic
			self.critic_optimizer.zero_grad ()
			critic_loss.backward ()
			self.critic_optimizer.step ()

			# Compute actor loss
			actor_loss = -self.critic ( state , self.actor ( state ) ).mean ()

			# Optimize the actor
			self.actor_optimizer.zero_grad ()
			actor_loss.backward ()
			self.actor_optimizer.step ()

			# Update the target networks
			for param , target_param in zip ( self.critic.parameters () , self.critic_target.parameters () ):
				target_param.data.copy_ ( soft_tau * param.data + (1 - soft_tau) * target_param.data )

			for param , target_param in zip ( self.actor.parameters () , self.actor_target.parameters () ):
				target_param.data.copy_ ( soft_tau * param.data + (1 - soft_tau) * target_param.data )
