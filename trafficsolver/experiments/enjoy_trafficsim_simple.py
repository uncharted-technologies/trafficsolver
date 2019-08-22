#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 10:18:48 2019

@author: maxime

"""
import pickle
import torch
import time

import matplotlib.pyplot as plt
import numpy as np

from algorithm.dqn import DQN
from algorithm.mlp import MLP
from environments.trafficsim_simple import TrafficSimSimple

env = TrafficSimSimple()

agent = DQN(env,MLP)

agent.load('network.pth')

obs = env.reset()
returns = 0
for i in range(10000):
    action = agent.predict(torch.FloatTensor(obs))
    obs, rew, done, info = env.step(action)
    env.render()
    time.sleep(0.5)
    returns += rew
    if done:
        obs = env.reset()
        print("Episode score: ",returns)
        returns = 0