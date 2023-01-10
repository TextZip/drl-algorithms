import gymnasium as gym 

env = gym.make("CliffWalking-v0",render_mode="rgb_array")
terminated = False
G = 0 # Return 
observation, info = env.reset(seed=42)
counter = 0
gamma = 1
while not terminated:
    action = env.action_space.sample() # random policy
    observation,reward,terminated,truncated,info = env.step(action=action)
    G += reward*(gamma**counter)
    counter +=1
env.close()
print(f"The episode terminated after {counter} steps with Return(G) {G} for gamma {gamma}")