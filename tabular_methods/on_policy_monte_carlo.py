import gymnasium as gym
import numpy as np
env = gym.make('FrozenLake-v1', desc=None, map_name="8x8",
               is_slippery=False, render_mode="human")