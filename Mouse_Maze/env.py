import numpy as np


class env:
    def __init__(self):
        self.action_max = 4
        self.state_max = 6
        self.one_chess = 1
        self.two_chess = 2
        self.big_chess = 10
        self.rat_poison = -10
        self.state = 0
        self.next_state = 0
        self.maze = self.build_env()

    '''
    0 : top
    1 : down
    2 : left
    3 : right
    '''
    def reset(self):
        self.state = 0
        return self.state
    def step(self, action):
        next_state = self.state

        if action == 0:
            if self.state - 2 < 0:
                pass
            else:
                next_state = self.state % 3
                # I should not work this is , but this step never do
        elif action == 1:
            if self.state - 2 > 0:
                pass
            else:
                next_state = self.state + 3
        elif action == 2:
            if self.state == 0 or self.state == 3:
                pass
            else:
                next_state = self.state - 1
        elif action == 3:
            if self.state == 2 or self.state == 5:
                pass
            else:
                next_state = self.state + 1

        temp_s = self.state
        self.state = next_state

        reward, done = self.check_reward(next_state)

        return temp_s, reward, next_state, done

    def check_reward(self,next_state):
        reward = 0; done =False
        if next_state == 1:
            reward +=1
        elif next_state == 3:
            reward +=2
        elif next_state == 4:
            reward -= 10; done =True
        elif next_state == 5:
            reward += 10; done = True

        return reward, done

    '''
           0|1|2
           3|4|5
    '''

    def build_env(self):
        maze = np.zeros(self.state_max)
        maze[1] = self.one_chess
        maze[3] = self.two_chess
        maze[4] = self.rat_poison
        maze[5] = self.big_chess

        return maze

    def show(self,s):
        print('███████')
        for i in range(0,3):
            print('|',end='')
            if i == s:
                print('X',end='')
            else:
                print(' ',end='')
        print('|')
        print('███████')
        for i in range(3,6):
            print('|', end='')
            if i == s:
                print('X', end='')
            else:
                print(' ', end='')
        print('|')
        print('███████')

