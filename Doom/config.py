# -*- coding: utf-8 -*-
"""
This file is config all need hyperparameters
"""
# scenarios path
config_file_path = "./scenarios/simpler_basic.cfg"
# model save
model_savefile = "./model-doom.pth"
#  images size
resolution = (30, 45)
# memory size
replay_memory_size = 10000
# learning hyperparameters
learning_rate = 0.00025
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
