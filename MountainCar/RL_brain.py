import numpy as np
import pandas as pd

ESPILON = 0.9 #using in greedy
ALPHA = 0.1 #learning rate
GAMMA = 0.9 #explore_rate



class QLearningTable:
    def __init__(self, n_states, actions, learning_rate=ALPHA, explore_rate=GAMMA, greedy=GAMMA):
        self.actions = actions
        self.n_states = n_states
        self.lr = learning_rate
        self.er = explore_rate
        self.epsilon = greedy
        self.build_q_table(self.n_states, self.actions)

    def build_q_table(self,n_states,actions):
        table = pd.DataFrame(
            np.zeros((n_states, len(actions))),
            columns= actions,
        )
        print(table)

        return table

