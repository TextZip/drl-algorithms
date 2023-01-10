import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt
from envs import Maze

env = Maze()
print(f"The action space is of the type: {env.action_space}")
print(f"The observation space is of the type: {env.observation_space}")
print(f"A random action is:{env.action_space.sample()}")
env.close()