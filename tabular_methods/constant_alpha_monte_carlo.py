import gymnasium as gym
import numpy as np
env = gym.make('FrozenLake-v1', desc=None, map_name="8x8",
               is_slippery=False, render_mode="rgb_array")

action_values = np.zeros((env.observation_space.n,env.action_space.n))

def policy(state,epsilon=0.2):
    if np.random.random()<epsilon:
        return np.random.choice(env.action_space.n)
    else:
        action_value = action_values[state]
        return np.random.choice(np.flatnonzero(action_value == action_value.max()))

def constant_alpha_monte_carlo(policy,action_values,episodes,gamma=0.99,epsilon=0.2,alpha=0.2):

    for episode in range(1, episodes+1):
        state,info = env.reset()
        done = False
        transitions = []

        while not done:
            action = policy(state=state,epsilon=epsilon)
            next_state,reward,terminated,truncated,info = env.step(action=action)
            done = terminated or truncated
            transitions.append([state,action,reward])
            state = next_state

        G = 0

        for state_t,action_t,reward_t in reversed(transitions):
            G = reward_t + gamma*G

            old_value = action_values[state_t,action_t]
            action_values[state_t,action_t] += alpha*(G-old_value)

    env.close()


constant_alpha_monte_carlo(policy=policy,action_values=action_values,episodes=20000)

env = gym.make('FrozenLake-v1', desc=None, map_name="8x8",
               is_slippery=False, render_mode="human")
observation,info = env.reset()
terminated = False
while not terminated:
    action = policy(observation,epsilon=0)
    observation, reward, terminated, truncated, info = env.step(action=action)
env.close()