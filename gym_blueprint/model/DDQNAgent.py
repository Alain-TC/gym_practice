# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf
from .replay_memory.memory import Memory
from .basic_agent import Agent
from .basic_model import SmallModel


class DDQNAgent(Agent):
    def __init__(self, state_size, action_size, epsilon_decay=0.99, memory_size=1000000, gamma=0.99, tau=.001):
        super().__init__(state_size, action_size, epsilon_decay, memory_size, gamma)
        self.USE_PER = True
        self.ddqn = True
        self.MEMORY = Memory(memory_size)
        self.tau = tau

        self.model = SmallModel(self.state_size, self.action_size, self.learning_rate, False)
        self.target_model = SmallModel(self.state_size, self.action_size, self.learning_rate, False)
        self.update_target_model(1)

    def update_target_model(self, tau):
        # copy weights from model to target_model        #print(self.model.get_weights()[0])
        new_weights = []
        for i in range(len(self.model.get_weights())):
            new_weights.append(tau * self.model.get_weights()[i] + (1 - tau) * self.target_model.get_weights()[i])
        self.target_model.set_weights(new_weights)

    def memorize(self, state, action, reward, next_state, done):
        experience = state, action, reward, next_state, done
        if self.USE_PER:
            self.MEMORY.store(experience)
        else:
            self.memory.append((experience))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        self.batch_size = batch_size
        # if len(self.memory) < self.train_start:
        #    return
        # Randomly sample minibatch from the memory
        if self.USE_PER:
            # Sample minibatch from the PER memory
            tree_idx, minibatch = self.MEMORY.sample(self.batch_size)
        else:
            # Randomly sample minibatch from the deque memory
            minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))

        state = np.zeros((self.batch_size, self.state_size))
        next_state = np.zeros((self.batch_size, self.state_size))
        action, reward, done = [], [], []

        # do this before prediction
        # for speedup, this could be done on the tensor level
        # but easier to understand using a loop
        for i in range(self.batch_size):
            state[i] = minibatch[i][0]
            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            next_state[i] = minibatch[i][3]
            done.append(minibatch[i][4])

        # do batch prediction to save speed
        target = self.model.predict(state)
        target_next = self.model.predict(next_state)
        target_val = self.target_model.predict(next_state)

        for i in range(len(minibatch)):
            # correction on the Q value for the action used
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                if self.ddqn:  # Double - DQN
                    # current Q Network selects the action
                    # a'_max = argmax_a' Q(s', a')
                    a = np.argmax(target_next[i])
                    # target Q Network evaluates the action
                    # Q_max = Q_target(s', a'_max)
                    target[i][action[i]] = reward[i] + self.gamma * (target_val[i][a])
                else:  # Standard - DQN
                    # DQN chooses the max Q value among next actions
                    # selection and evaluation of action is on the target Q Network
                    # Q_max = max_a' Q_target(s', a')
                    target[i][action[i]] = reward[i] + self.gamma * (np.amax(target_next[i]))

                    # Train the Neural Network with batches
            target_old = np.array(target)

            if self.USE_PER:
                absolute_errors = np.abs(target_old[i] - target[i])
                # Update priority
                self.MEMORY.batch_update(tree_idx, absolute_errors)

        # Train the Neural Network with batches
        self.model.fit(state, target, batch_size=self.batch_size, verbose=0)
        self.update_target_model(self.tau)
