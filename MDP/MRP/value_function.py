import numpy as np
'''
This is solution of Student Markov Chain by value function
'''

# state of dictionary
d = {'c1': 0, 'c2': 1, 'c3': 2, 'ps': 3, 'pub': 4, 'fb': 5, 'sp': 6}
gamma = 1  # discount factor
episode_time = 100  # run times
node_num = 7  # the student state
'''build MDP question
class1 : 0
class2 : 1
class3 : 2
pass   : 3
pub    : 4
face   : 5
sleep  : 6

Returns:
    [type] -- [description]
    numpy -- this is state,action matrix
'''


def env_build():
    matrix_p = np.zeros([node_num, node_num])
    matrix_r = np.zeros([node_num])
    matrix_s = np.ones([node_num])
    # C1 ->Fb;->C2
    matrix_p[d['c1']][d['c2']] = 0.5
    matrix_p[d['c1']][d['fb']] = 0.5
    matrix_r[d['c1']] = -2
    # C2 ->Sp;->C3
    matrix_p[d['c2']][d['sp']] = 0.2
    matrix_p[d['c2']][d['c3']] = 0.8
    matrix_r[d['c2']] = -2
    # C3 ->Pub;Ps
    matrix_p[d['c3']][d['pub']] = 0.4
    matrix_p[d['c3']][d['ps']] = 0.6
    matrix_r[d['c3']] = -2
    # Ps ->Sp;
    matrix_p[d['ps']][d['sp']] = 1
    matrix_r[d['ps']] = 10
    # Pub ->C1;->C2;->C3
    matrix_p[d['pub']][d['c1']] = 0.2
    matrix_p[d['pub']][d['c2']] = 0.4
    matrix_p[d['pub']][d['c3']] = 0.4
    matrix_r[d['pub']] = 1
    # Fb ->Fb;->C1
    matrix_p[d['fb']][d['fb']] = 0.9
    matrix_p[d['fb']][d['c1']] = 0.1
    matrix_r[d['fb']] = -1

    return matrix_s, matrix_r, matrix_p


'''
    show result
Returns:
    none
'''


def show_result(s):
    print('c1 ->', s[0])
    print('c2 ->', s[1])
    print('c3 ->', s[2])
    print('pass ->', s[3])
    print('pub ->', s[4])
    print('fb ->', s[5])
    print('sleep ->', s[6])


'''value iteration
    calcuate value iteration
Returns:
    [type] -- [description]
    state_  update value
'''


def value_iteration(state, s, r, p):
    # print('count ', state)
    vs = 0
    # immediately reward
    vi = r[state]
    # print('immediately reward = ', vi)
    # expect reward
    for i in range(0, node_num):
        vs += (p[state][i] * s[i])
    state_ = vi + (gamma * vs)

    return state_


def value_iteration_new(s, r, p):
    s_ = r + (gamma * np.dot(p, s))

    return s_


'''update matrix
    update state
Returns:
    [type] -- [description]
    nparry -- return state set
'''


def update_matrix(state_, index, s):

    s[index] = state_
    return s


'''Do episode
    define how times you want do
'''


def episode(times):
    s, r, p = env_build()
    print('init matrix s = ', s)
    print('init matrix r = ', r)
    print('init matrix p = \n', p)

    for i in range(0, times):
        for j in range(0, node_num):
            state_ = value_iteration(j, s, r, p)
            s = update_matrix(state_, j, s)
        print('iteration ~', i)
        print('after update = ', s)

    return (s)


def episode_new(times):
    s, r, p = env_build()

    for i in range(0, times):
        s_ = value_iteration_new(s, r, p)
        s = s_
        print('iteration ~', i)
    return (s)


if __name__ == '__main__':

    s = episode_new(episode_time)
    show_result(s)
