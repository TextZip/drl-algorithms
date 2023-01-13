import gymnasium as gym
import numpy as np
env = gym.make('FrozenLake-v1', desc=None, map_name="8x8",
                is_slippery=False, render_mode="rgb_array")

action_values = np.zeros((env.observation_space.n,env.action_space.n))

def target_policy(state):
    action_value = action_values[state]
    return np.random.choice(np.flatnonzero(action_value == action_value.max()))

def exploratory_policy(state,epsilon=0.2):
    if np.random.random()<epsilon:
        return np.random.choice(env.action_space.n)
    else:
        action_value = action_values[state]
        return np.random.choice(np.flatnonzero(action_value == action_value.max()))


def off_policy_monte_carlo(action_values,target_policy,exploratory_ploicy,episodes,gamma=0.99,epsilon=0.2):
    counter_sa_values = np.zeros((env.observation_space.n,env.action_space.n))

    for episode in range(1,episodes+1):
        state,_ = env.reset()
        done = False
        transitions = [] 

        while not done:
            action = exploratory_ploicy(state=state,epsilon=epsilon)
            next_state,reward,terminated,truncated,_ = env.step(action=action)
            done = terminated or truncated
            transitions.append([state,action,reward])
            state = next_state
        
        G = 0 
        W = 1     

        for state_t,action_t,reward_t in reversed(transitions):
            G = reward_t + gamma*G
            counter_sa_values[state_t,action_t] += W
            old_value = action_values[state_t,action_t]
            action_values[state_t,action_t] += (W/counter_sa_values[state_t,action_t])* (G - old_value)

            if action_t != target_policy(state_t):
                break

            W = W*(1/(1-epsilon + (epsilon/4)))
    env.close()

off_policy_monte_carlo(action_values=action_values,target_policy=target_policy,exploratory_ploicy=exploratory_policy,episodes=5000)

env = gym.make('FrozenLake-v1', desc=None, map_name="8x8",
               is_slippery=False, render_mode="human")
observation,_ = env.reset()
terminated = False
while not terminated:
    action = target_policy(observation)
    observation, reward, terminated, truncated, info = env.step(action=action)
env.close()