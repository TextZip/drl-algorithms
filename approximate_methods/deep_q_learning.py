import random
import copy
import gymnasium as gym
import torch 
import torch.nn.functional as F
from torch import nn as nn
import matplotlib.pyplot as plt
from torch.optim import Adam
from tqdm import tqdm

env = gym.make("CartPole-v1",render_mode="rgb_array")

state_dims = env.observation_space.shape[0]
num_actions = env.action_space.n

def policy(state, epsilon=0.):    
    if torch.rand(1) < epsilon:
        return torch.randint(num_actions, (1, 1))
    else:
        av = q_network(state).detach()
        return torch.argmax(av, dim=-1, keepdim=True)

class PreprocessEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    # env.reset - > [N x D] 
    def reset(self):
        state,_ = self.env.reset()
        return torch.from_numpy(state).unsqueeze(dim=0).float()
    
    # env.step
    def step(self,action):
        action = action.item()
        next_state,reward,termination,truncation,_ = self.env.step(action=action)
        done = termination or truncation
        next_state = torch.from_numpy(next_state).unsqueeze(dim=0).float()
        reward = torch.tensor(reward).view(1,-1).float()
        done = torch.tensor(done).view(1, -1)

        return next_state, reward, done, _

class Q_network(nn.Module):
    def __init__(self,state_dim,action_dim):
        super(Q_network,self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features=state_dim,out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128,out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64,out_features=action_dim),
        )

    def forward(self,x):
        return self.block(x)

class ReplayMemory():
    def __init__(self,capacity=1000000):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def insert(self,transition):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition
        self.position = (self.position+1) % self.capacity

    def sample(self,batch_size):
        assert self.can_sample(batch_size)
        batch = random.sample(self.memory, batch_size) #[[s,a,r,d,s'],[s,a,r,d,s']] --> [[s,s,s],[a,a,a],[r,r,r],[d,d,d],[s',s',s']]
        batch = zip(*batch)
        return [torch.cat(items) for items in batch] # output size is NxD

    def can_sample(self,batch_size):
        return len(self.memory) >= batch_size*10

    def __len__(self):
        return len(self.memory)

def deep_q_learning(q_network,policy,episodes,alpha=0.001,batch_size=32,gamma=0.99,epsilon=0.2):
    optimizer = Adam(params=q_network.parameters(),lr=alpha)
    memory = ReplayMemory(capacity=1000000)
    stats = {'MSE Loss':[], 'Returns': []}

    for episode in tqdm(range(1,episodes+1)):
        state = env.reset()
        done = False
        ep_return = 0

        while not done:
            action = policy(state=state,epsilon=epsilon)
            next_state,reward,done,_ = env.step(action=action)
            memory.insert([state,action,reward,done,next_state])

            if memory.can_sample(batch_size=batch_size):
                state_b,action_b,reward_b,done_b,next_state_b = memory.sample(batch_size=batch_size)
                action_value_b = q_network(state_b).gather(1,action_b)
                next_action_value_b = q_network(next_state_b)
                next_action_value_b = torch.max(next_action_value_b, dim =1, keepdim=True)[0]

                target_b = reward_b + ~done_b* gamma * next_action_value_b
                loss = F.mse_loss(action_value_b, target_b)
                q_network.zero_grad()
                loss.backward()
                optimizer.step()

                stats['MSE Loss'].append(loss.item())
            state = next_state
            ep_return += reward.item()
        stats['Returns'].append(ep_return)
        if episode % 10 ==0:
            target_q_network.load_state_dict(q_network.state_dict())
    return stats

q_network = Q_network(state_dim=state_dims,action_dim=num_actions)
target_q_network = copy.deepcopy(q_network)
target_q_network = target_q_network.eval()

env = PreprocessEnv(env=env)
stats = deep_q_learning(q_network=q_network,policy=policy,episodes=300,epsilon=0.01)
# print(stats)

env = gym.make("CartPole-v1",render_mode="human")
env = PreprocessEnv(env=env)
while True:
    observation = env.reset()
    terminated = False
    while not terminated:
        action = policy(observation, epsilon=0)
        observation, reward, done, info = env.step(action=action)
    env.close()
