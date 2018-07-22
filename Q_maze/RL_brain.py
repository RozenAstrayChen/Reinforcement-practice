import numpy as np
import pandas as pd

log = "QLearningTable: "


class QLearningTable:
    #%%
    #init
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
    """[summary]
    choose action
    """
    #%%
    def choose_action(self, observation):
        self.check_state_exist(observation)

        #action
        if np.random.uniform() < self.epsilon: # normal
            state_action = self.q_table.loc[observation, :]


            # The same state,
            # there may be multiple identical Q action values
            # so we random it
            state_action = state_action.reindex(
                np.random.permutation(state_action.index))
            action = state_action.argmax()

        else: #random select action
            action = np.random.choice(self.actions)

        return  action
    #%%
    """[summary]
    learning update params
    """
    def learn(self, s, a, r, s_):
        self.check_state_exist(s_) #check q_table alive s_
        q_predict = self.q_table.loc[s,a]

        if s_ != 'terminal':
            #Import algorithm
            q_target = r +\
                       self.gamma * \
                       self.q_table.loc[s_, :].max() #if next state not terminal
        else:
            q_target = r #next state is terminal
        self.q_table.loc[s, a] += self.lr *\
                                  (q_target - q_predict) #update state-action vlaue
    #%%
    """[summary]
    check state alive
    """
    def check_state_exist(self, state):
        if state not in self.q_table.index:
            #append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index = self.q_table.columns,
                    name = state,
                )
            )

    #%%
