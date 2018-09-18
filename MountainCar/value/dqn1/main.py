import gym
import dqn_improve as dqn
import numpy as np
''' env size
state -> 24(nparray)
action -> 3 element
    1. action = 24(nparray)
    2. reward = 1(float)
    3. done = 1(bool)
'''

trials = 100
trial_len = 500
updateTargetNetwork = 1000
TRAIN_MODE = True
TEST_MODE = False
model_name = 'my_model.h5'
'hyperparamter in position'
alpha = 10


def reward_count(b_velocity,position,velocity):    
    position = abs(position - (-0.5))
    doubleAS = abs(velocity**2 - b_velocity**2)
    acceleration = (doubleAS / position)/2
    print('pos =', position, 'acc = ', acceleration )
    reward = acceleration * position
    return reward

if __name__ == "__main__":
    # env = gym.make('BipedalWalker-v2')
    env = gym.make('MountainCar-v0')
    state_size = env.observation_space.shape[0]
    env.reset()  # Cannot call env.step() before calling reset()
    # action_size = len(env.action_space.sample())
    action_size = env.action_space.n
    # Init agent
    agent = dqn.DQNAgent(env, state_size, action_size)
    print(agent.model.summary())

    if TRAIN_MODE:
        for trial in range(trials):
            cur_state = env.reset().reshape(1,2)
            # setting velocity is zero
            b_velocity = 0
            for step in range(trial_len):
                env.render()
                action = agent.act(cur_state)
                new_state, reward, done, _ = env.step(action)

                # find acc
                position, velocity = new_state
                reward = reward_count(b_velocity, position*alpha, velocity)
                print(reward)

                # reward = reward if not done else -20
                new_state = new_state.reshape(1,2)

                agent.remember(cur_state, action, reward, new_state, done)
                agent.replay()       # internally iterates default (prediction) model
                agent.target_train() # iterates target model

                cur_state = new_state
                b_velocity = velocity


                if done:
                    break

            if step >= 199:
                print("Failed to complete in trial {}".format(trial))
                if step % 10 == 0:
                    agent.save_model("trial-{}.model".format(trial))
            else:
                print("Completed in {} trials".format(trial))
                agent.save_model(model_name)
                #break

        #agent.save("./save/cartpole-dqn.h5")
    if TEST_MODE:
        agent.load(model_name)
        agent.epsilon = agent.epsilon_min
        for trial in range(trials):
            cur_state = env.reset().reshape(1, 2)
            for step in range(trial_len):
                action = agent.act(cur_state)
                env.render()
                new_state, reward, done, _ = env.step(action)
                new_state = new_state.reshape(1, 2)
            
                cur_state = new_state
                if done:
                    break
            print("epsoide is ", trial)
            if step >= 299:
                print("Failed to complete trial")    
            else:
                print("Completed in {} trials".format(trial))

