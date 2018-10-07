# -*- coding: utf-8 -*-
"""
This file is config all need hyperparameters
"""
# scenarios path
map_basic = "./scenarios/simpler_basic.cfg"
map_corridor = "./scenarios/deadly_corridor.cfg"
map_d_line = "./scenarios/defend_the_line.cfg"
map_d_center = "./scenarios/defend_the_center.cfg"
map_health = "./scenarios/health_gathering.cfg"
map_xhealth = "./scenarios/health_gathering_supreme.cfg"
map_match = "./scenarios/deathmatch.cfg"
map_mwh = "./scenarios/my_way_home.cfg"
map_cover = "./scenarios/take_cover.cfg"
map_oblige = "./scenarios/oblige.cfg"
# model save
savefile = ".pth"
modle2_savefile = "./model4-doom.pth"
#  images size
'''
in first basic wad i using 30,45
in second map, i using origin pixel size
'''
resolution = (60, 108)
# memory size
replay_memory_size = 10000
# learning hyperparameters
learning_rate = 0.00025
policy_lr = 0.002
gamma = 0.99
epsilon = 1
dec_eps = 0.995
min_eps = 0.01
epochs = 20
learning_step_per_epoch = 2000
test_step_per_epoch = 100
watch_step_per_epoch = 10
# NN
batch_size = 64
# frame
frame_repeat = 12

# parameters
deep_q_netowrk = 'dqn'
policy_gradient = 'pg'
actor_cirtic = 'ac'
double_dqn = 'ddqn'
