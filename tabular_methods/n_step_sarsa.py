import gymnasium as gym
import numpy as np

env = gym.make('FrozenLake-v1', desc=None, map_name="8x8",
               is_slippery=False, render_mode="rgb_array")

action_values = np.zeros((env.observation_space.n, env.action_space.n))


def policy(state, epsilon=0.2):
    if np.random.random() < epsilon:
        return np.random.choice(env.action_space.n)
    else:
        action_value = action_values[state]
        return np.random.choice(np.flatnonzero(action_value == action_value.max()))


# n-step-on-policy-td-learning
def n_step_sarsa(action_values, episodes, policy, alpha=0.1, gamma=0.99, epsilon=0.2, n=1):
    for episode in range(episodes+1):
        state, _ = env.reset()
        action = policy(state=state, epsilon=epsilon)
        transitions = []
        done = False
        t = 0

        while t-n < len(transitions):
            if not done:
                next_state, reward, termination, truncation, _ = env.step(
                    action=action)
                done = termination or truncation
                next_action = policy(state=next_state, epsilon=epsilon)
                transitions.append([state, action, reward])

            if t >= n:
                G = (1-done)*action_values[state, action]
                for state_t, action_t, reward_t in reversed(transitions[t-n:]):
                    G = reward_t + gamma*G

                action_values[state_t, action_t] += alpha * \
                    (G - action_values[state_t, action_t])
            t += 1
            state = next_state
            action = next_action
    env.close()


n_step_sarsa(action_values=action_values, episodes=1000, policy=policy)

env = gym.make('FrozenLake-v1', desc=None, map_name="8x8",
               is_slippery=False, render_mode="human")
observation, _ = env.reset()
terminated = False
while not terminated:
    action = policy(observation, epsilon=0)
    observation, reward, terminated, truncated, info = env.step(action=action)
env.close()
