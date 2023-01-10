import gymnasium as gym

env = gym.make("CliffWalking-v0",render_mode="rgb_array")

observation,info = env.reset(seed=42)
terminated = False
trajectory = []
while not terminated:
    action = env.action_space.sample()
    new_observation,reward,terminated,truncated,info = env.step(action=action)
    trajectory.append([observation,action,reward])
    observation = new_observation
env.close()
print(f"Trajectory for entire episode: {trajectory}")
