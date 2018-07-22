
from Q_maze.maze_env import Maze
from Sarsa_maze.RL_brain import QLearningTable


#%%
def update():
    #learing 100 time
    for episode in range(1):
        #init state
        observation = env.reset()

        while True:
            #update visualization enviroment
            env.render()

            #The RL brain selects action based on observation
            # of the state
            action = RL.choose_action(str(observation))

            #Do action by exploer,
            # and got next state observation ,reward and done
            # by enviroment
            observation_, reward, done = env.step(action)
            print("observation_ = ",observation_)
            print("reward = ",reward)
            print("action = ",action)
            print("done = ",done)

            #RL learn in sequenece(state, action, reward, state_)
            RL.learn(str(observation), action, reward, str(observation_))

            #Sending next state to next episode
            observation = observation_

            #If fall into the hole or  get goal then end this round
            if done:
                break

    # left game and close windows
    print('game over')
    env.destroy()
#%%

if __name__ == "__main__":
    # define env and RL function
    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)))

    # start visualzation env
    env.after(100, update)
    env.mainloop()
    
#%%