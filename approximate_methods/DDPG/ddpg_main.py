import gymnasium as gym
import numpy as np
from ddpg_agent import Agent
import matplotlib.pyplot as plt


def plot_learning_curves(scores, x, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0,i-100):(i+1)])
    plt.plot(x,running_avg)
    plt.title('Running avg of 100 scores')
    plt.savefig(figure_file)


if __name__ == '__main__':
    env = gym.make('LunarLanderContinuous-v2')
    # print([env.observation_space.shape[0]])
    agent = Agent(0.0001,0.001,input_dims=env.observation_space.shape,
                tau=0.001,batch_size=64,fc1_dims=400,fc2_dims=300,
                n_actions=env.action_space.shape[0])
    
    n_games = 1000
    filename = 'LunarLander_alpha_' + str(agent.alpha) + '_beta_' + str(agent.beta) + '_' + str(n_games) + '_games'
    figure_file = 'plots/' + filename + '.png'

    best_score = env.reward_range[0]
    score_history = []
    for i in range(n_games):
        observation,_ = env.reset()
        done = False
        score = 0
        agent.noise.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done1,done2, info = env.step(action)
            done = done1 or done2
            agent.remember(observation,action,reward,observation_,done)
            agent.learn()
            score += reward
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()
        
        print(f"episode: {i} | score: {score:.2f} | avg_score: {avg_score:.2f}")

    x = [i+1 for i in range(len(n_games))]
    plot_learning_curves(score_history,x,figure_file=figure_file)