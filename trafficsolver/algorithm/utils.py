#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 10:53:23 2019

@author: william
"""

import random

import numpy as np
import torch


def set_global_seed(seed, env):
    torch.manual_seed(seed)
    env.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
