from ac import *
from dqn import *
from ddqn import *
from policy import *
import argparse
from config import *

parser = argparse.ArgumentParser(description='choice Reinforcement model')
parser.add_argument('--map', type=str, default=map_basic, help='choice map')
parser.add_argument(
    '--model',
    type=str,
    default='dqn',
    help='choice reinforcement model(default is dqn)')
parser.add_argument('--load', type=int, default=0, help='load weight')
parser.add_argument('--load_num', type=int, default=0, help='loaditerator num')
parser.add_argument('--train', type=int, default=1, help='train_model')
parser.add_argument('--train_num', type=int, default=1, help='iterator time')
# watch type
parser.add_argument('--watch', type=int, default=0, help='watch_result')
parser.add_argument(
    '--watch_num', type=int, default=1, help='you want watch iterator num')

arg = parser.parse_args()
print(arg.model)
if arg.model == deep_q_netowrk:
    model = DQN(map=arg.map)
    print('using dqn!')
elif arg.model == double_dqn:
    model = DDQN(map=arg.map)
    print('using ddqn!')
elif arg.model == policy_gradient:
    model = Policy(arg.map)
    print('using pg!')
elif arg.model == actor_cirtic:
    model = AC(map_xhealth)
    print('using ac!')

if arg.train == 1:
    print('training-type')
    model.train_model(load=arg.load, num=arg.load_num, iterators=arg.train_num)
if arg.watch == 1:
    print('watch-type')
    model.watch_model(arg.watch_num)
