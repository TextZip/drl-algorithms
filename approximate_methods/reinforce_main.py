import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt 
from reinforce_agent import PolicyGradientAgent

def plot_learning_curves(scores, x, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0,i-100):(i+1)])
    plt.plot(x,running_avg)
    plt.title('Running avg of 100 scores')
    plt.savefig(figure_file)


if __name__  == '__main__':
    env = gym.make('LunarLander-v2')
    n_games = 3000
    agent = PolicyGradientAgent(gamma=0.99, lr=0.0005, input_dims=[8], n_actions=4)

    fname = 'REINFORCE_' + 'lunar_lander_' + str(agent.lr) + '_' + str(n_games) + 'games'
    figure_file = 'plots/' + fname + '.png'

    scores = []
    for i in range(n_games):
        done = False
        observation,_ = env.reset()
        score = 0 
        while not done:
            action = agent.choose_action(observation=observation)
            observation_,reward,done1,done2,info = env.step(action=action)
            done = done1 or done2
            score += reward
            agent.store_rewards(reward)
            observation = observation_
        agent.learn()
        scores.append(score)

        avg_score = np.mean(scores[-100:])
        print(f"episode: {i} score: {score:.2f} avg_score: {avg_score:.2f}")

    x = [i+1 for i in range(len(scores))]
    plot_learning_curves(scores,x,figure_file=figure_file)