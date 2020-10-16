import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import numpy as np
from collections import OrderedDict
from tensorboardX import SummaryWriter
from torch.distributions import Normal

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()

        setattr(self, "actor_l1", nn.Linear(state_dim, 16))
        setattr(self, "actor_l2", nn.Linear(16, 16))
        setattr(self, "actor_l3_mu", nn.Linear(16, action_dim))
        setattr(self, "actor_l3_std", nn.Linear(16, action_dim))

        self.critic = nn.Sequential(
            nn.Linear(state_dim, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, 1))

    def forward(self, x):
        params = OrderedDict(self.named_parameters())

        x = F.linear(x, weight=params["actor_l1.weight"], bias=params["actor_l1.bias"])
        x = F.relu(x)
        x = F.linear(x, weight=params["actor_l2.weight"], bias=params["actor_l2.bias"])
        x = F.relu(x)
        mu = F.linear(x, weight=params["actor_l3_mu.weight"], bias=params["actor_l3_mu.bias"])
        std = F.linear(x, weight=params["actor_l3_std.weight"], bias=params["actor_l3_std.bias"])
        return mu, std

    def act(self, state, memory):
        mu, std = self(state)
        dist = Normal(mu, F.softplus(std) + 1e-5)
        action = dist.sample()
        action_logprob = torch.sum(dist.log_prob(action), dim=-1)
        
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)
        
        return action.detach()
    
    def evaluate(self, state, action):   
        mu, std = self(state)
        dist = Normal(mu, F.softplus(std) + 1e-5)
        action_logprobs = torch.sum(dist.log_prob(action), dim=-1)
        dist_entropy = torch.sum(dist.entropy(), dim=-1)
        state_value = self.critic(state)
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        
        self.policy_old = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
    
    def select_action(self, state, memory):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.policy_old.act(state, memory).cpu().data.numpy().flatten()
    
    def update(self, memory):
        # Monte Carlo estimate of rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # convert list to tensor
        old_states = torch.squeeze(torch.stack(memory.states).to(device), 1).detach()
        old_actions = torch.squeeze(torch.stack(memory.actions).to(device), 1).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs), 1).to(device).detach()
        
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        

def main():
    # Hyperparameters
    env_name = "pointmass-v0"
    horizon = 64
    render = False
    log_interval = 20        # print avg reward in the interval
    max_episodes = 100000    # max training episodes
    update_timestep = 20     # update policy every n timesteps
    K_epochs = 50            # update policy for K epochs
    eps_clip = 0.2           # clip parameter for PPO
    gamma = 0.99             # discount factor
    lr = 0.0003              # parameters for Adam optimizer
    betas = (0.9, 0.999)
    random_seed = None

    # Set logging
    if not os.path.exists("./logs"):
        os.makedirs("./logs")
    tb_writer = SummaryWriter('./logs/tb_log')

    # creating environment
    import gym_env
    env = gym.make(env_name)
    env._max_episode_steps = horizon
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    if random_seed:
        print("Random Seed: {}".format(random_seed))
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)
    
    memory = Memory()
    ppo = PPO(state_dim, action_dim, lr, betas, gamma, K_epochs, eps_clip)
    
    # logging variables
    running_reward = 0
    avg_length = 0
    time_step = 0
    
    # training loop
    for i_episode in range(1, max_episodes + 1):
        state = env.reset()

        for t in range(horizon):
            time_step += 1

            # Running policy_old:
            action = ppo.select_action(state, memory)
            state, reward, done, _ = env.step(action)
            
            # Saving reward and is_terminals:
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            
            # update if its time
            if time_step % update_timestep == 0:
                ppo.update(memory)
                memory.clear_memory()
                time_step = 0

            running_reward += reward * pow(gamma, t)
            if render:
                env.render()
            if done:
                break
        
        avg_length += t
        
        # save every 500 episodes
        if i_episode % 500 == 0:
            torch.save(ppo.policy.state_dict(), './PPO_continuous_{}.pth'.format(env_name))
            
        # logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length / log_interval)
            running_reward = int((running_reward / log_interval))
            
            print('Episode {} \t Avg length: {} \t Avg reward: {}'.format(i_episode, avg_length, running_reward))
            tb_writer.add_scalar("reward", running_reward, i_episode)
            running_reward = 0
            avg_length = 0
            

if __name__ == '__main__':
    main()
