# -*- coding: utf-8 -*-
from vizdoom import *
import numpy as np
from config import *


def init_doom(scenarios=map_basic, visable=False):
    print('Initializing doom...')
    game = DoomGame()
    game.load_config(scenarios)
    game.set_window_visible(visable)
    game.set_mode(Mode.PLAYER)
    # game.set_screen_format(ScreenFormat.GRAY8)
    game.set_screen_format(ScreenFormat.RGB24)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    # 06
    game.set_doom_map('map01')
    game.init()
    print('Doom initialized')
    return game


def test_env(game):
    print('Test start')
    temp_epochs = 10
    for epoch in range(temp_epochs):
        game.new_episode()
        while not game.is_episode_finished():
            pass
    game.close()
