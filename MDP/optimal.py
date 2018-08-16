import numpy as np

d = {'c1': 0, 'c2': 1, 'c3': 2, 'sp': 3, 'fb': 4}
gamma = 1
v_pi = 0.5
node_size = 5
episode_time = 100


def value_iterator(s):
    # c1
    c1a1 = v_pi * (-1 + s[d['fb']] * gamma)
    c2a2 = v_pi * (-2 + s[d['c2']] * gamma)
    s[d['c1']] = c1a1 + c2a2

    # c2
    c2a1 = v_pi * (0 + s[d['sp']] * gamma)
    c2a2 = v_pi * (-2 + s[d['c3']] * gamma)
    s[d['c2']] = c2a1 + c2a2

    # c3
    c3a1 = v_pi * (10 + s[d['sp']] * gamma)
    c3a2 = v_pi * (1 + 0.2 * s[d['c1']] * gamma + 0.4 * s[d['c2']] * gamma +
                   0.4 * s[d['c3']] * gamma)
    s[d['c3']] = c3a1 + c3a2

    # fb
    fba1 = v_pi * (-1 + s[d['fb']] * gamma)
    fba2 = v_pi * (0 + s[d['c1']] * gamma)
    s[d['fb']] = fba1 + fba2

    return s


def show_result(s):
    print('c1 ->', s[0])
    print('c2 ->', s[1])
    print('c3 ->', s[2])
    print('pass ->', s[3])
    print('fb ->', s[4])


def epsoide(times):
    s = np.zeros(node_size)
    for i in range(0, times):
        s_ = value_iterator(s)
        s = s_
        show_result(s)

    print('iterator over!')

    return s


if __name__ == '__main__':

    s = epsoide(episode_time)
    show_result(s)
