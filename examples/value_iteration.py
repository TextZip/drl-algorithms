import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v1', desc=None, map_name="8x8",is_slippery=False, render_mode="human")

observation,info = env.reset()
terminated = False

policy_probs = np.full((env.observation_space.n,env.action_space.n),0.25) # state-action table
state_values = np.zeros(shape=(env.observation_space.n))

def value_iteration(policy_probs,state_values,theta=1e-6,gamma=0.99):
    delta = 1
    while delta>theta:
        delta=0
        for state in range(env.observation_space.n):
            old_value = state_values[state]
            action_values = np.zeros(env.action_space.n)
            max_q = float('-inf')
            for action in range(env.action_space.n):
                probablity, next_state, reward, info = env.P[state][action][0]
                action_values[action] = probablity*(reward + (gamma*state_values[next_state]))
                if action_values[action] > max_q:
                    max_q = action_values[action]
                    action_probs = np.zeros(env.action_space.n)
                    action_probs[action]  = 1
            state_values[state] = max_q
            policy_probs[state] = action_probs 

            delta = max (delta, abs(old_value - state_values[state]))

def policy(state):
    return np.argmax(policy_probs[state])

value_iteration(policy_probs=policy_probs,state_values=state_values)

while not terminated:
    action = policy(observation)
    observation,reward,terminated,truncated,info = env.step(action=action)
env.close()
