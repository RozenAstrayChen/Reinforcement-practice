import gym
import pandas as pd
import matplotlib.pyplot as plt
from MountainCar.value.tensor_dqn import DeepQNetwork

env = gym.make('MountainCar-v0')
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

RL = DeepQNetwork(n_actions=3, n_features=2, learning_rate=0.001, e_greedy=0.9,
                  replace_target_iter=300, memory_size=3000,
                  e_greedy_increment=0.0001, )

total_steps = 0
positions = []
episodes = 1000
steps = 200


def car_final_position():
    import time
    localtime = time.asctime(time.localtime(time.time()))
    localtime = localtime+  '.png'
    plt.figure(2, figsize=[10, 5])
    p = pd.Series(positions)
    ma = p.rolling(10).mean()
    plt.plot(p, alpha=0.8)
    plt.plot(ma)
    plt.xlabel('Epsiode')
    plt.ylabel('Poistion')
    plt.title('Car Final Position - Modified.png')
    plt.show()
    plt.savefig(localtime)

def reward_judge(position):
    reward = 0
    if position > 0.1:
        reward =  1
    elif position > 0.25:
        reward =  2
    elif position > 0.5:
        reward =  10
    return reward

for i in range(0, episodes):
    # env.render()
    observation = env.reset()
    ep_r = 0
    for s in range(steps):
        action = RL.choose_action(observation)

        observation_, reward, done, info = env.step(action)

        position, velocity = observation_
        positions.append(position)
        # 车开得越高 reward 越大
        reward = abs(position - (-0.5))
        reward +=  reward_judge(position)


        RL.store_transition(observation, action, reward, observation_)

        if total_steps > 1000:
            RL.learn()

        ep_r += reward
        if done:
            get = '| Get' if observation_[0] >= env.unwrapped.goal_position else '| ----'
            print('Epi: ', i,
                  get,
                  '| Ep_r: ', round(ep_r, 4),
                  '| Epsilon: ', round(RL.epsilon, 2))
            break
        if s >= 199:
            print('episoide', i, ' is failed')
            print('reward is ', ep_r)
            break

        observation = observation_
        total_steps += 1
# RL.save()
RL.plot_cost()
car_final_position()
