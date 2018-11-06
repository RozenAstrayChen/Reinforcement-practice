from tensorflow.python import keras as ks
import numpy as np
import gym
import Model


class Agent:
    def __init__(self, env_name='MountainCar-v0'):
        self.env = gym.make(env_name)
        states = self.env.observation_space.shape[0]
        actions = self.env.action_space.n
        self.batch_size = 32
        self.agent = Model.DQNAgent(states, actions)

    def train(self):
        episodes = 1000
        for e in range(episodes):
            state = self.env.reset()
            state = np.reshape(state, [1, 2])
            rewards = 0
            for time_t in range(500):
                # self.env.render()
                action = self.agent.choose(state)
                print(action)
                state_, reward, done, _ = self.env.step(action)

                if state_[0] >= 0.1:
                    reward += 10
                elif state_[0] >= 0.25:
                    reward += 20
                elif state_[0] >= 0.5:
                    reward += 100

                rewards += reward

                state_ = np.reshape(state_, [1, 2])
                self.agent.transition(state, action, reward, state_, done)

                state = state_

                if time_t >= 199:
                    print('episode: {}/{} failed!'.format(e, episodes))
                    break
                if done and time_t <= 199:
                    print('episode: {}/{}, successful!'.format(
                        e, episodes, rewards))
                    
                if len(self.agent.memory) > self.batch_size:
                    self.agent.replay(self.batch_size)
            print('episode: {}/{}, score: {}'.format(e, episodes, rewards))

        # finish
        self.final_position(self.agent.actions)
        self.final_reward(rewards)

    def final_position(self, positions):
        import time
        localtime = time.localtime()
        timeString = time.strftime('%m%d%H', localtime)
        timeString = str(timeString) + '.jpg'

        plt.figure(3, figsize=[10, 5])
        p = pd.Series(positions)
        ma = p.rolling(10).mean()
        plt.plot(p, alpha=0.8)
        plt.plot(ma)
        plt.xlabel('Epsiode')
        plt.ylabel('Position')
        plt.title('Final Position')
        plt.show()

    def final_reward(self, rewards):
        import time
        localtime = time.localtime()
        timeString = time.strftime('%m%d%H', localtime)
        timeString = 'position' + str(timeString) + '.jpg'

        plt.figure(1, figsize=[10, 5])
        p = pd.Series(reward)
        ma = p.rolling(10).mean()
        plt.plot(p, alpha=0.8)
        plt.plot(ma)
        plt.xlabel('Epsiode')
        plt.ylabel('Reward')
        plt.title('Final Rewards')
        plt.show()
