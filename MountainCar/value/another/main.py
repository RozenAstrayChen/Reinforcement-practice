
import gym
import MountainCar.value.another.Model as nn
import MountainCar.value.another.GameRunner as agent
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

def car_final_position(positions):
    import time
    localtime = time.localtime()
    timeString = time.strftime("%m%d%H", localtime)
    timeString = str(timeString) + '.jpg'

    plt.figure(2, figsize=[10, 5])
    p = pd.Series(positions)
    ma = p.rolling(10).mean()
    plt.plot(p, alpha=0.8)
    plt.plot(ma)
    plt.xlabel('Epsiode')
    plt.ylabel('Poistion')
    plt.title('Car Final Position - Modified.png')
    plt.savefig(timeString)
    plt.show()


if __name__ == "__main__":
    env_name = 'MountainCar-v0'
    env = gym.make(env_name)

    num_states = env.env.observation_space.shape[0]
    num_actions = env.env.action_space.n

    model =  nn.Model(num_states, num_actions, 50)
    mem = nn.Memory(50000)

    with tf.Session() as sess:
        sess.run(model._var_init)
        gr = agent.GameRunner(sess, model, env, mem, 0.9, 0.1,
                        0.9)
        num_episodes = 300
        cnt = 0
        while cnt < num_episodes:
            if cnt % 10 == 0:
                print('Episode {} of {}'.format(cnt+1, num_episodes))
            gr.run()
            cnt += 1
        plt.plot(gr._reward_store)
        plt.show()
        plt.close("all")
        plt.plot(gr._max_x_store)
        plt.show()
        car_final_position(gr._max_x_store)