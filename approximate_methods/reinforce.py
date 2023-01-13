import random
import copy
import numpy as np
import gymnasium as gym
import torch 
import torch.nn.functional as F
from torch import nn as nn
import matplotlib.pyplot as plt
from torch.optim import Adam
from tqdm import tqdm
import os

num_envs = os.cpu_count()
env = gym.vector.make('CartPole-v1',num_envs=num_envs)
dims = env.observation_space.shape[1]
num_actions = env.single_action_space.n

class PreprocessEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self):
        state,info = self.env.reset()
        return torch.from_numpy(state).float()

    def step(self,actions):
        actions = actions.squeeze()
        next_state,reward,termination,truncation,info = self.env.step(actions=actions)
        done = torch.logical_or(torch.tensor(termination),torch.tensor(truncation)).view(-1,1)
        next_state = torch.from_numpy(next_state).float()
        reward = torch.tensor(reward).view(-1,1).float()
        return next_state,reward,done,info

policy = nn.Sequential(
    nn.Linear(in_features=dims,out_features=128),
    nn.ReLU(),
    nn.Linear(in_features=128,out_features=64),
    nn.ReLU(),
    nn.Linear(in_features=64,out_features=num_actions),
    nn.Softmax(dim=1)
)

# def reinforce(policy, episodes, alpha=1e-4,gamma=0.99):
#     optimizer = Adam(params=policy.parameters(),lr=alpha)
#     stats = {'Loss': [], 'Returns': []}

#     for episode in tqdm(range(1,episodes+1)):
#         state = env.reset()
#         done_b = torch.zeros((num_envs,1),dtype=torch.bool)
#         transitions = []
#         ep_return = torch.zeros((num_envs,1))

#         while not done_b.all():
#             action = policy(state).multinomial(1).detach()
#             next_state,reward,done,_ = env.step(actions=action)
#             transitions.append([state,action,~done_b*reward])
#             ep_returns += reward
#             done_b |= done
        
#         G = np.zeros((num_envs,1))

#         for t, (state_t,action_t,reward_t) in reversed(list(enumerate(transitions))):
#             G = reward_t + gamma*G
#             probs_t = policy(state_t)
#             log_probs_t = torch.log(probs_t + 1e-6)
#             action_log_prob_t = log_probs_t.gather(1,action_t)

#             entropy_t = - torch.sum(probs_t*log_probs_t, dim=-1,keepdim=True)
#             gamma_t = gamma **t
#             pg_loss_t = -gamma_t * action_log_prob_t * G
#             total_loss_t = (pg_loss_t - 0.01*entropy_t).mean()

#             policy.zero_grad()
#             total_loss_t.backward()
#             optimizer.step()
#         stats['Loss'].append(total_loss_t.item())
#         stats['Returns'].append(ep_return.mean().item())
#     # env.close()
#     return stats

def reinforce(policy, episodes, alpha=1e-4, gamma=0.99):
    optim = Adam(policy.parameters(), lr=alpha)
    stats = {'PG Loss': [], 'Returns': []}
    
    for episode in tqdm(range(1, episodes + 1)):
        state = env.reset()
        done_b = torch.zeros((num_envs, 1), dtype=torch.bool)
        transitions = []
        ep_return = torch.zeros((num_envs, 1))

        while not done_b.all():
            action = policy(state).multinomial(1).detach()
            next_state, reward, done, _ = env.step(action)
            transitions.append([state, action, ~done_b * reward])
            ep_return += reward
            done_b |= done
            state = next_state
        
        G = torch.zeros((num_envs, 1))
        for t, (state_t, action_t, reward_t) in reversed(list(enumerate(transitions))):
            G = reward_t + gamma * G
            probs_t = policy(state_t)
            log_probs_t = torch.log(probs_t + 1e-6)
            action_log_prob_t = log_probs_t.gather(1, action_t)

            entropy_t = - torch.sum(probs_t * log_probs_t, dim=-1, keepdim=True)
            gamma_t = gamma ** t
            pg_loss_t = - gamma_t * action_log_prob_t * G
            total_loss_t = (pg_loss_t - 0.01 * entropy_t).mean()
            
            policy.zero_grad()
            total_loss_t.backward()
            optim.step()

        stats['PG Loss'].append(total_loss_t.item())
        stats['Returns'].append(ep_return.mean().item())

    return stats

env = PreprocessEnv(env=env)
stats = reinforce(policy=policy,episodes=200)
env = gym.vector.make('CartPole-v1',num_envs=num_envs)
env = PreprocessEnv(env=env)
state = env.reset()
