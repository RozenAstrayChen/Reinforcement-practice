import dqn
import gym
import numpy as np

episodes = 500
EPISODES = 100
batch_size =32 
TRAIN_MODE = True
TEST_MODE = False

if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    print(state_size,action_size)
    agent = dqn.DQNAgent(state_size,action_size)
    if TRAIN_MODE:
        #Iterate the game
        for e in range(EPISODES):

            #rest state in the beginning of each game
            state = env.reset()
            state = np.reshape(state, [1,4])

            #time_t represents each frame of the game
            for time_t in  range(500):
                # turn this on if you want to render
                # env.render()
                # Decide action
                action = agent.act(state)
                # Advance the game to the next frame based on the action.
                # Reward is 1 for every frame the pole survived
                next_state, reward, done, _ = env.step(action)
                next_state = np.reshape(next_state, [1, 4])
                # Remember the previous state, action, reward, and done
                agent.remember(state, action, reward, next_state, done)
                # make next_state the new current state for the next frame.
                state = next_state
                # done becomes True when the game ends
                # ex) The agent drops the pole
                if done:
                    # print the score and break out of the loop
                    print("episode: {}/{}, score: {}"
                        .format(e, 100, time_t))
                    break
                if len(agent.memory) > batch_size:
                    # train the agent with the experience of the episode
                    agent.replay(batch_size)
        agent.save("./save/cartpole-dqn.h5")
    
    if TEST_MODE:
        agent.load('./save/cartpole-dqn.h5')
        for e in range(EPISODES):
            #rest state in the beginning of each game
            state = env.reset()
            state = np.reshape(state, [1,4])
            for time_t in range(episodes):
                env.render()
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                next_state = np.reshape(next_state, [1, 4])
                # Remember the previous state, action, reward, and done
                agent.remember(state, action, reward, next_state, done)
                # make next_state the new current state for the next frame.
                state = next_state
                # done becomes True when the game ends
                # ex) The agent drops the pole
                if done:
                    # print the score and break out of the loop
                    print("episode: {}/{}, score: {}"
                        .format(e, EPISODES, time_t))
                    break