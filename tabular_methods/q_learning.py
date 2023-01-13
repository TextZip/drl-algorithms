import gymnasium as gym
import numpy as np
env = gym.make('FrozenLake-v1', desc=None, map_name="8x8",
               is_slippery=False, render_mode="rgb_array")

action_values = np.zeros((env.observation_space.n, env.action_space.n))


def target_policy(state):
    action_value = action_values[state]
    return np.random.choice(np.flatnonzero(action_value == action_value.max()))


def exploratory_policy():
    return np.random.randint(env.action_space.n)


def q_learning(action_values, episodes, target_policy, exploratory_policy, alpha=0.1, gamma=0.99):  # on-policy-td-learning
    for episode in range(episodes+1):
        state, _ = env.reset()
        done = False

        while not done:
            action = exploratory_policy()
            next_state, reward, termination, truncation, _ = env.step(
                action=action)
            done = termination or truncation
            next_action = target_policy(state=next_state)

            action_value = action_values[state, action]
            next_action_value = action_values[next_state, next_action]
            action_values[state, action] = action_value + alpha * \
                (reward + gamma*next_action_value - action_value)
            state = next_state
    env.close()

q_learning(action_values=action_values, episodes=5000, target_policy=target_policy, exploratory_policy=exploratory_policy, alpha=0.1, gamma=0.99)

env = gym.make('FrozenLake-v1', desc=None, map_name="8x8",
               is_slippery=False, render_mode="human")
observation, _ = env.reset()
terminated = False

while not terminated:
    action = target_policy(observation)
    observation, reward, terminated, truncated, info = env.step(action=action)
env.close()
