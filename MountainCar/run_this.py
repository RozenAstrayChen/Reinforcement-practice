import gym
import numpy as np
import random
import math
from time import sleep
from MountainCar.RL_brain import QLearningTable

NUM_EPISODES = 1000
MAX_T = 200
DEBUG_MODE = True
POSITION_MAX = 0.6
ACTIONS = ['left','neutral','right']

#Initialize the  enviroment
env = gym.make('MountainCar-v0')



'''
    Observation has two variables
    [position, velocity] 
'''
def update_function_test():
    for episode in range(NUM_EPISODES):
        #Reset the enviroment
        obv = env.reset()


        for t in range(MAX_T):
            env.render()
            #action will got four of array
            action = env.action_space.sample()

            obv, reward, done, info = env.step(0)

            # Print data
            if (DEBUG_MODE):
                print("\nEpisode = %d" % episode)
                print("observation =", obv)
                print("done =",done)
                print("info = ",info)
                print("Action: " , action)
                #print("State: %s" % str(state))
                print("Reward: %f" % reward)

                print("")







if __name__ == '__main__':
    RL = QLearningTable(MAX_T,ACTIONS)
    update_function_test()


