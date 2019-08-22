#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ted on Thu Jan  3 2019

@author: maxime
"""
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from algorithm.replay_buffer import ReplayBuffer
from algorithm.logger import Logger
from algorithm.utils import set_global_seed


class DQN:
    def __init__(
        self,
        env,
        network,
        replay_start_size=50000,
        replay_buffer_size=1000000,
        gamma=0.99,
        update_target_frequency=10000,
        minibatch_size=32,
        learning_rate=1e-3,
        update_frequency=1,
        initial_exploration_rate=1,
        final_exploration_rate=0.1,
        final_exploration_step=1000000,
        adam_epsilon=1e-8,
        logging=False,
        log_folder=None,
        seed=None,
        loss="mse",
    ):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.replay_start_size = replay_start_size
        self.replay_buffer_size = replay_buffer_size
        self.gamma = gamma
        self.update_target_frequency = update_target_frequency
        self.minibatch_size = minibatch_size
        self.learning_rate = learning_rate
        self.update_frequency = update_frequency
        self.initial_exploration_rate = initial_exploration_rate
        self.epsilon = self.initial_exploration_rate
        self.final_exploration_rate = final_exploration_rate
        self.final_exploration_step = final_exploration_step
        self.adam_epsilon = adam_epsilon
        self.logging = logging
        self.log_folder = log_folder
        if callable(loss):
            self.loss = loss
        else:
            try:
                self.loss = {'huber': F.smooth_l1_loss, 'mse': F.mse_loss}[loss]
            except KeyError:
                raise ValueError("loss must be 'huber', 'mse' or a callable")

        self.env = env
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)
        self.seed = random.randint(0, 1e6) if seed is None else seed
        set_global_seed(self.seed, self.env)

        self.network = network(self.env.observation_space, self.env.action_space.n).to(self.device)
        self.target_network = network(self.env.observation_space, self.env.action_space.n).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate, eps=self.adam_epsilon)

    def learn(self, timesteps, verbose=False):

        if self.logging:
            logger = Logger(self.log_folder)

        # On initialise l'état
        state = torch.as_tensor(self.env.reset())
        score = 0
        t1 = time.time()

        for timestep in range(timesteps):

            is_training_ready = timestep >= self.replay_start_size

            # On prend une action
            action = self.act(state.to(self.device).float(), is_training_ready=is_training_ready)

            # Mise à jour d'epsilon
            self.update_epsilon(timestep)

            # On execute l'action dans l'environnement
            state_next, reward, done, _ = self.env.step(action)

            # On stock la transition dans le replay buffer
            action = torch.as_tensor([action], dtype=torch.long)
            reward = torch.as_tensor([reward], dtype=torch.float)
            done = torch.as_tensor([done], dtype=torch.float)
            state_next = torch.as_tensor(state_next)
            self.replay_buffer.add(state, action, reward, state_next, done)

            score += reward.item()

            if done:
                # On réinitialise l'état
                if verbose:
                    print("Timestep : {}, score : {}, Time : {} s".format(timestep, score, round(time.time() - t1, 3)))
                if self.logging:
                    logger.add_scalar('Episode_score', score, timestep)
                state = torch.as_tensor(self.env.reset())
                score = 0
                t1 = time.time()
            else:
                state = state_next

            if is_training_ready:

                # Update du réseau principal
                if timestep % self.update_frequency == 0:

                    # On sample un minibatche de transitions
                    transitions = self.replay_buffer.sample(self.minibatch_size, self.device)

                    # On s'entraine sur les transitions selectionnées
                    loss = self.train_step(transitions)
                    if self.logging:
                        logger.add_scalar('Loss', loss, timestep)

                # Si c'est le moment, on update le target Q network on copiant les poids du network
                if timestep % self.update_target_frequency == 0:
                    self.target_network.load_state_dict(self.network.state_dict())

        if self.logging:
            logger.save()

    def train_step(self, transitions):
        # huber = torch.nn.SmoothL1Loss()
        states, actions, rewards, states_next, dones = transitions

        # Calcul de la Q value via le target network (celle qui maximise)
        with torch.no_grad():
            q_value_target = self.target_network(states_next.float()).max(1, True)[0]

        # Calcul de la TD Target
        td_target = rewards + (1 - dones) * self.gamma * q_value_target

        # Calcul de la Q value en fonction de l'action jouée
        q_value = self.network(states.float()).gather(1, actions)

        loss = self.loss(q_value, td_target, reduction='mean')

        # Update des poids
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def act(self, state, is_training_ready=True):
        if is_training_ready and random.uniform(0, 1) >= self.epsilon:
            # Action qui maximise la Q function
            action = self.predict(state)
        else:
            # Action aléatoire ( sur l'interval [a, b[ )
            action = np.random.randint(0, self.env.action_space.n)
        return action

    def update_epsilon(self, timestep):
        eps = self.initial_exploration_rate - (self.initial_exploration_rate - self.final_exploration_rate) * (
            timestep / self.final_exploration_step
        )
        self.epsilon = max(eps, self.final_exploration_rate)

    @torch.no_grad()
    def predict(self, state):
        action = self.network(state).argmax().item()
        return action

    def save(self,timestep=None):
        if timestep is not None:
            filename = 'network_' + str(timestep) + '.pth'
        else:
            filename = 'network.pth'

        if self.log_folder is not None:
            save_path = self.log_folder + '/' +filename
        else:
            save_path = filename

        torch.save(self.network.state_dict(), save_path)

    def load(self,path):
        self.network.load_state_dict(torch.load(path,map_location='cpu'))
