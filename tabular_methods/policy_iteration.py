import gymnasium as gym
import numpy as np

env = gym.make('FrozenLake-v1', desc=None, map_name="8x8",
               is_slippery=False, render_mode="human")
observation, info = env.reset()
terminated = False

# state-action table
policy_probs = np.full((env.observation_space.n, env.action_space.n), 0.25)
state_values = np.zeros(shape=(env.observation_space.n))


def policy_iteration(policy_probs, state_values, theta=1e-6, gamma=0.99):
    policy_stable = False
    while policy_stable == False:
        # policy eval
        delta = 1
        while delta > theta:
            delta = 0
            for state in range(env.observation_space.n):
                old_value = state_values[state]
                new_state_value = 0
                for action in range(env.action_space.n):
                    probablity, next_state, reward, info = env.P[state][action][0]
                    new_state_value += policy_probs[state, action]*(
                        reward + (gamma*state_values[next_state]))
                state_values[state] = new_state_value
                delta = max(delta, abs(old_value-state_values[state]))
        # policy improvement
        policy_stable = True
        for state in range(env.observation_space.n):
            old_action = policy(state=state)
            action_values = np.zeros(env.action_space.n)
            max_q = float('-inf')
            for action in range(env.action_space.n):
                probablity, next_state, reward, info = env.P[state][action][0]
                action_values[action] = probablity * \
                    (reward + (gamma*state_values[next_state]))
                if action_values[action] > max_q:
                    max_q = action_values[action]
                    action_probs = np.zeros(env.action_space.n)
                    action_probs[action] = 1
            policy_probs[state] = action_probs
        # check termination condition and update policy_stable variable
            if old_action != policy(state=state):
                policy_stable = False


def policy(state):
    return np.argmax(policy_probs[state])


policy_iteration(policy_probs=policy_probs, state_values=state_values)
print(policy_probs)
while not terminated:
    action = policy(observation)
    observation, reward, terminated, truncated, info = env.step(action=action)
env.close()
