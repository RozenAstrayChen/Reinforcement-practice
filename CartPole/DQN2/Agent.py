import gym
from CartPole.DQN2.Model import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class Agent:
    def __init__(self,env_name='CartPole-v0'):
        self.env = gym.make(env_name)
        self.batch_size = 32
        num_states = self.env.observation_space.shape[0]
        num_actions = self.env.action_space.n

        self.agent = DQNAgent(num_states, num_actions)

    def train(self):
        # initialize gym environment and the agent
        episodes = 1000
        # Iterate the game
        for e in range(episodes):
            # reset state in the beginning of each game
            state = self.env.reset()
            state = np.reshape(state, [1, 4])
            rewards = 0
            # time_t represents each frame of the game
            # Our goal is to keep the pole upright as long as possible until score of 500
            # the more time_t the more score
            for time_t in range(500):

                # turn this on if you want to render
                # env.render()
                # Decide action
                action = self.agent.choose(state)
                # Advance the game to the next frame based on the action.
                # Reward is 1 for every frame the pole survived
                next_state, reward, done, _ = self.env.step(action)
                rewards += reward

                next_state = np.reshape(next_state, [1, 4])
                # Remember the previous state, action, reward, and done
                self.agent.transition(state, action, rewards, next_state, done)
                # make next_state the new current state for the next frame.
                state = next_state
                
                # done becomes True when the game ends
                # ex) The agent drops the pole
                
                if done:
                    # print the score and break out of the loop
                    print("episode: {}/{}, score: {}"
                          .format(e, episodes, time_t))
                    break
                if len(self.agent.memory) > self.batch_size:
                    self.agent.replay(self.batch_size)
                
            # train the agent with the experience of the episode
            print("episode: {}/{}, score: {}"
                  .format(e, episodes, rewards))
            
        self.final_position(self.agent.actions)
        self.agent.save()
    def test(self):
        # initialize gym environment and the agent
        self.agent.load()
        for i in range(10):
            state = self.env.reset()

            state = np.reshape(state, [1, 4])
            for j in range(500):
                self.env.render()
                action = self.agent.choose(state)
                # Advance the game to the next frame based on the action.
                # Reward is 1 for every frame the pole survived
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, 4])

                state = next_state

                
                if done:
                    # print the score and break out of the loop
                    print("game over")
                    break
                

    def final_position(self,positions):
        import time
        localtime = time.localtime()
        timeString = time.strftime("%m%d%H", localtime)
        timeString = str(timeString) + '.jpg'

        plt.figure(3, figsize=[10, 5])
        p = pd.Series(positions)
        ma = p.rolling(10).mean()
        plt.plot(p, alpha=0.8)
        plt.plot(ma)
        plt.xlabel('Epsiode')
        plt.ylabel('Poistion')
        plt.title('Final Position - Modified.jpg')
        plt.savefig(timeString)
        plt.show()





