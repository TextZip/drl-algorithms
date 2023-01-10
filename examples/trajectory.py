import gymnasium as gym

env = gym.make("CliffWalking-v0", render_mode="human")

observation, info = env.reset(seed=42)
trajectory = []

for _ in range(5):
    action = env.action_space.sample()
    new_observation,reward,terminated,truncated,info = env.step(action=action)
    trajectory.append([observation,action,reward])
    observation=new_observation
    if terminated or truncated:
        observation, info = env.reset()
env.close()
print(f"The Trajectory for 5 steps is: {trajectory}")