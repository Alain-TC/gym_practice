# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K
from keras.models import model_from_json
import tensorflow as tf
from .replay_memory.memory import Memory


EPISODES = 5000

class Agent:
    def __init__(self):
        self.target_model = None

    def to_json(self):
        return self.target_model.to_json()

    def save_weights(self, path):
        return self.target_model.save_weights(path)

class DDQNAgent(Agent):
    def __init__(self, state_size, action_size, epsilon_decay=0.99, memory_size=1000000, gamma=0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.USE_PER = True
        self.ddqn = True
        self.MEMORY = Memory(memory_size)

        self.memory = deque(maxlen=memory_size)

        self.gamma = gamma    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.001
        self.epsilon_decay = epsilon_decay
        self.learning_rate = 0.001
        self.tau = .001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model(1)
        self.target_counter = 0

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    """Huber loss for Q Learning
    References: https://en.wikipedia.org/wiki/Huber_loss
                https://www.tensorflow.org/api_docs/python/tf/losses/huber_loss
    """
    def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond  = K.abs(error) <= clip_delta

        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)

        return K.mean(tf.where(cond, squared_loss, quadratic_loss))

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(Dense(64, input_dim=64, activation='relu'))
        model.add(Dense(32, input_dim=64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss=self._huber_loss,
                      optimizer=Adam(lr=self.learning_rate))
        return model

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

    # def replay(self, batch_size):
    #     minibatch = random.sample(self.memory, batch_size)
    #     for state, action, reward, next_state, done in minibatch:
    #         target = self.model.predict(state)
    #         target[0][action] = reward + [self.gamma * np.amax(self.target_model.predict(next_state)[0]), 0][done]
    #         self.model.fit(state, target, epochs=1, verbose=0)
    #     self.update_target_model(self.tau)

    def replay(self, batch_size):
        self.batch_size = batch_size
        #if len(self.memory) < self.train_start:
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


    def load(self, name):
        self.model.load_weights(name)
        print("Loaded model weights")

    def save(self, name):
        self.model.save_weights(name)

    def save_json(self, json_path):
        model_json = self.model.to_json()
        with open(json_path, "w") as json_file:
            json_file.write(model_json)

# load json and create model
    def load_model(self, json_path):
        json_file = open(json_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        print("Loaded model architecture")
