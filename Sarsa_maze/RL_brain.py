from Q_maze.RL_brain import QLearningTable

class SarsaTable(QLearningTable):

    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        super(SarsaTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)

    def learn(self, s, a, r, s_, a_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, a_]  # q_target is choose the best a_ not greedy of  Q(s_)
        else:
            q_target = r  # if s is terminal.
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # 更新 q_table
