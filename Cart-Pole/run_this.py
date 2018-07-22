import gym
import numpy as np
import random
import math
from time import sleep

#Initialize the "Cart-Pole" enviroment
env = gym.make('CartPole-v0')


#Defining the environment related constants

# Number of discrete states (bucket)` per state dimension
NUM_BUCKETS = (1, 1, 6, 3)
# Number of discrete actions
NUM_ACTIONS = env.action_space.n #(left, right)
#Bounds for each discrete state
STATE_BOUNDS = list(zip(env.observation_space.low,
                        env.observation_space.high
                        ))
STATE_BOUNDS[1] = [-0.5, 0.5]
STATE_BOUNDS[3] = [-math.radians(50),math.radians(50)]
#Index of action
ACTION_INDEX = len(NUM_BUCKETS)

## Creating a Q-Table for each state-action pair
## (1, 1, 6, 3 + 2)
q_table = np.zeros(NUM_BUCKETS + (NUM_ACTIONS,))

## Learning related constants
MIN_EXPLORE_RATE = 0.01
MIN_LEARNING_RATE = 0.1

##Defining the simulation related constants
NUM_EPISODES = 1000
MAX_T = 250
STREAK_TO_END = 120
SOLVED_T = 199
#SOLVED_T = 1000
DEBUG_MODE = True


def read_module(name):
    # load...
    load_table = np.load(name)

    return load_table

def record_rate(episode,rate):
    fp = open("explore_rate.txt", "a")
    string = str(episode) + ":" + str(rate) + "\n"
    fp.write(string)
    fp.close()

"""[Summary]
    Simulate
"""
def train_simulate(module):

    learn_table = module

    ## Instantiating the learning related parameters

    learning_rate = get_learning_rate(0)
    explore_rate = get_explore_rate(0)
    discount_factor = 0.99  # since the world is unchanging

    num_streaks = 0

    for episode in range(NUM_EPISODES):

        #Reset the enviroment
        obv = env.reset()

        #the initial state
        state_0 = state_to_bucket(obv)

        for t in range(MAX_T):
            env.render()

            #Select an action
            action = select_action(state_0, explore_rate, learn_table)

            #Execute the action
            obv, reward, done, _ = env.step(action)

            #Observe the result
            state = state_to_bucket(obv)

            #Update the Q based on the result
            best_q = np.amax(q_table[state])
            #Algorithm,
            # l_rate( reward + Predict - Reality )
            learn_table[state_0 + (action,)] += learning_rate * (reward + discount_factor*(best_q) - learn_table[state_0 + (action,)])


            #Setting up for the next iteration
            state_0 = state

            # Print data
            if (DEBUG_MODE):
                print("\nEpisode = %d" % episode)
                print("observation = ", obv)
                print("Action: %d" % action)
                print("State: %s" % str(state))
                print("Reward: %f" % reward)
                print("Best Q: %f" % best_q)
                print("Explore rate: %f" % explore_rate)
                print("Learning rate: %f" % learning_rate)
                print("Streaks: %d" % num_streaks)

                print("")



        if int(episode) is 10:
            np.save('10times.npy',learn_table)
            record_rate(10, explore_rate)
            sleep(0.25)

        elif  int(episode) is 100:
            np.save('100times.npy', learn_table)
            record_rate(100, explore_rate)
            sleep(0.25)

        elif int(episode) == 200:
            np.save('200times.npy', learn_table)
            record_rate(200, explore_rate)
            sleep(0.25)

        elif int(episode) == 300:
            np.save('300times.npy', learn_table)
            record_rate(300, explore_rate)
            sleep(0.25)

        elif int(episode) == 350:
            np.save('350times.npy', learn_table)
            record_rate(350, explore_rate)
            sleep(0.25)

        elif int(episode) == 400:
            np.save('400times.npy', learn_table)
            record_rate(400, explore_rate)
            sleep(0.25)

        elif int(episode) == 450:
            np.save('450times.npy', learn_table)
            record_rate(450, explore_rate)
            sleep(0.25)

        elif int(episode) == 500:
            np.save('500times.npy', learn_table)
            record_rate(500, explore_rate)
            sleep(0.25)


        # Update parameters
        explore_rate = get_explore_rate(episode)
        learning_rate = get_learning_rate(episode)



def test_simulate(module,explore_rate):

    testing_table = module

    ## Instantiating the learning related parameters


    for episode in range(NUM_EPISODES):
        #Reset the enviroment
        obv = env.reset()

        #the initial state
        state_0 = state_to_bucket(obv)

        for t in range(MAX_T):
            env.render()

            #Select_action
            action = select_action(state_0, explore_rate,testing_table)

            #Execute the action
            obv, _, _, _ = env.step(action)

            #Observe the result
            state = state_to_bucket(obv)

            # Setting up for the next iteration
            state_0 = state

            # Print data
            if (DEBUG_MODE):
                print("\nEpisode = %d" % episode)
                print("observation = ", obv)
                print("Action: %d" % action)
                print("State: %s" % str(state))
                print("Explore rate: %f" % explore_rate)

                print("")






def select_action(state, explore_rate,table):
    # Select a random action
    if random.random() < explore_rate:
        action = env.action_space.sample()
    # Select the action with the highest q
    else:
        action = np.argmax(table[state])
    return action


def get_explore_rate(t):
    return max(MIN_EXPLORE_RATE, min(1, 1.0 - math.log10((t+1)/25)))

def get_learning_rate(t):
    return max(MIN_LEARNING_RATE, min(0.5, 1.0 - math.log10((t+1)/25)))

def state_to_bucket(state):
    bucket_indice = []
    for i in range(len(state)):
        if state[i] <= STATE_BOUNDS[i][0]:
            bucket_index = 0
        elif state[i] >= STATE_BOUNDS[i][1]:
            bucket_index = NUM_BUCKETS[i] - 1
        else:
            # Mapping the state bounds to the bucket array
            bound_width = STATE_BOUNDS[i][1] - STATE_BOUNDS[i][0]
            offset = (NUM_BUCKETS[i]-1)*STATE_BOUNDS[i][0]/bound_width
            scaling = (NUM_BUCKETS[i]-1)/bound_width
            bucket_index = int(round(scaling*state[i] - offset))
        bucket_indice.append(bucket_index)
    return tuple(bucket_indice)

if __name__ == "__main__":


    #train_simulate(q_table)

    test_simulate(read_module('500times.npy'),0.1)
    #obv = env.reset()
    #print(obv)


    #print(read_module('100times.npy'))



