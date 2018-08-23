import Mouse_Maze.agent as agent

ag = agent.Agent()
episodes = 100
steps = 6

for episode in range(episodes):
    state = ag.env.reset()
    for step in range(steps):
        action = ag.choose(state)
        state,reward,state_,done = ag.env.step(action)
        print('-----action = ', action, '-----------')
        print('state = ', state)
        print('reward = ', reward)
        print('next_state = ', state_)
        print('done = ', done)

        ag.learning(state, action, reward, state_)
        print(ag.table)
        if done:
            print('-------------game over--------------')
            break
        state_ = state
    print('----episode over next ',episode)

