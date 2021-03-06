import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import threading
import gym
import multiprocessing
import numpy as np
from queue import Queue
import argparse
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import layers
from .recording import record
from keras.utils import plot_model
from .basic_agent import Agent

tf.enable_eager_execution()

'''
Adapted from https://github.com/tensorflow/models/blob/master/research/a3c_blogpost/a3c_cartpole.py
'''


class ActorCriticModel(keras.Model):
    def __init__(self, state_size, action_size):
        super(ActorCriticModel, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.dense1 = layers.Dense(128, activation='relu')
        self.dense3 = layers.Dense(128, activation='relu')
        self.policy_logits = layers.Dense(action_size)
        self.dense2 = layers.Dense(128, activation='relu')
        self.dense4 = layers.Dense(128, activation='relu')
        self.values = layers.Dense(1)

    def call(self, inputs):
        # Forward pass
        x = self.dense1(inputs)
        x3 = self.dense3(x)
        logits = self.policy_logits(x3)
        v1 = self.dense2(inputs)
        v4 = self.dense4(v1)
        values = self.values(v4)
        return logits, values


class MasterAgent(Agent):
    def __init__(self, game_name, state_size, action_size, lr=.001):
        self.lr = lr
        self.game_name = game_name
        self.state_size = state_size
        self.action_size = action_size
        self.opt = tf.train.AdamOptimizer(self.lr, use_locking=True)

        self.global_model = ActorCriticModel(self.state_size, self.action_size)  # global network
        self.global_model(tf.convert_to_tensor(np.random.random((1, self.state_size)), dtype=tf.float32))
        print("summary")
        self.global_model.summary()
        plot_model(self.global_model, to_file='model_tmtc.png')

    def train(self):
        res_queue = Queue()

        workers = [Worker(self.state_size,
                          self.action_size,
                          self.global_model,
                          self.opt, res_queue,
                          i, game_name=self.game_name) for i in range(multiprocessing.cpu_count())]
        # i, game_name = self.game_name) for i in range(1)]

        for i, worker in enumerate(workers):
            print("Starting worker {}".format(i))
            worker.start()

        moving_average_rewards = []  # record episode reward to plot
        while True:
            reward = res_queue.get()
            if reward is not None:
                moving_average_rewards.append(reward)
            else:
                break
        [w.join() for w in workers]

        plt.plot(moving_average_rewards)
        plt.ylabel('Moving average ep reward')
        plt.xlabel('Step')
        # plt.savefig(os.path.join(self.save_dir,
        #                         '{} Moving Average.png'.format(self.game_name)))
        plt.show()

    def test(self, total_episode, render="True", name="A3C_test"):
        self.scores, self.episodes, self.average = [], [], []
        self.env = gym.make(self.game_name)
        mem = Memory()
        global_episode = 0
        while global_episode < total_episode:
            current_state = self.env.reset()
            mem.clear()
            ep_reward = 0.
            ep_steps = 0
            self.ep_loss = 0

            done = False
            while not done:  # and ep_steps<500:
                logits, _ = self.global_model(
                    tf.convert_to_tensor(current_state[None, :],
                                         dtype=tf.float32))
                probs = tf.nn.softmax(logits)

                action = np.random.choice(self.action_size, p=probs.numpy()[0])
                new_state, reward, done, _ = self.env.step(action)
                if render:
                    self.env.render()
                if done:
                    reward = -1
                ep_reward += reward
                mem.store(current_state, action, reward)

                if done:  # done and print information
                    self.plotModel(score=ep_reward, episode=global_episode, name=name)
                    global_episode += 1
                    print(ep_steps)

                ep_steps += 1
                current_state = new_state


class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def store(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []


class Worker(threading.Thread, Agent):
    # Temp
    # Set up global variables across different threads
    global_episode = 0
    # Moving average reward
    global_moving_average_reward = 0
    best_score = 0
    save_lock = threading.Lock()

    def __init__(self, state_size, action_size, global_model, opt, result_queue, idx, game_name,
                 save_dir='/tmp', max_episodes=500):
        super(Worker, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.result_queue = result_queue
        self.global_model = global_model
        self.opt = opt
        self.local_model = ActorCriticModel(self.state_size, self.action_size)
        self.worker_idx = idx
        self.game_name = game_name
        self.env = gym.make(self.game_name).unwrapped
        self.save_dir = save_dir
        self.ep_loss = 0.0
        self.max_episodes = max_episodes
        self.gamma = 0.99

        # temp
        self.scores, self.episodes, self.average = [], [], []

    def run(self, update_freq=20):
        total_step = 1
        mem = Memory()
        while Worker.global_episode < self.max_episodes:
            current_state = self.env.reset()
            mem.clear()
            ep_reward = 0.
            ep_steps = 0
            self.ep_loss = 0

            time_count = 0
            done = False
            count = 0
            while not done and count<500:
                count += 1
                logits, _ = self.local_model(
                    tf.convert_to_tensor(current_state[None, :],
                                         dtype=tf.float32))
                probs = tf.nn.softmax(logits)

                action = np.random.choice(self.action_size, p=probs.numpy()[0])
                new_state, reward, done, _ = self.env.step(action)
                if done:
                    reward = -1
                ep_reward += reward
                mem.store(current_state, action, reward)

                if time_count == update_freq or done:
                    # Calculate gradient wrt to local model. We do so by tracking the
                    # variables involved in computing the loss by using tf.GradientTape
                    with tf.GradientTape() as tape:
                        total_loss = self.compute_loss(done,
                                                       new_state,
                                                       mem,
                                                       self.gamma)
                    self.ep_loss += total_loss
                    # Calculate local gradients
                    grads = tape.gradient(total_loss, self.local_model.trainable_weights)
                    # Push local gradients to global model
                    self.opt.apply_gradients(zip(grads,
                                                 self.global_model.trainable_weights))
                    # Update local model with new weights
                    self.local_model.set_weights(self.global_model.get_weights())

                    mem.clear()
                    time_count = 0

                    if done:  # done and print information
                        self.plotModel(score=ep_reward, episode=Worker.global_episode, name="actor_3_critic")

                        Worker.global_moving_average_reward = \
                            record(Worker.global_episode, ep_reward, self.worker_idx,
                                   Worker.global_moving_average_reward, self.result_queue,
                                   self.ep_loss, ep_steps)
                        # We must use a lock to save our model and to print to prevent data races.
                        if ep_reward > Worker.best_score:
                            with Worker.save_lock:
                                print("Saving best model to {}, "
                                      "episode score: {}".format(self.save_dir, ep_reward))
                                self.global_model.save_weights(
                                    os.path.join(self.save_dir,
                                                 'model_{}.h5'.format(self.game_name))
                                )
                                Worker.best_score = ep_reward
                        Worker.global_episode += 1
                ep_steps += 1

                time_count += 1
                current_state = new_state
                total_step += 1
        self.result_queue.put(None)

    def compute_loss(self,
                     done,
                     new_state,
                     memory,
                     gamma=0.99):
        if done:
            reward_sum = 0.  # terminal
        else:
            reward_sum = self.local_model(
                tf.convert_to_tensor(new_state[None, :],
                                     dtype=tf.float32))[-1].numpy()[0]

        # Get discounted rewards
        discounted_rewards = []
        for reward in memory.rewards[::-1]:  # reverse buffer r
            reward_sum = reward + gamma * reward_sum
            discounted_rewards.append(reward_sum)
        discounted_rewards.reverse()

        logits, values = self.local_model(
            tf.convert_to_tensor(np.vstack(memory.states),
                                 dtype=tf.float32))
        r = tf.reshape(tf.convert_to_tensor(np.array(discounted_rewards)[:, None], dtype=tf.float32), values.shape)
        advantage = r - values

        # Value loss
        value_loss = advantage ** 2

        # Calculate our policy loss
        policy = tf.nn.softmax(logits)
        entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=policy, logits=logits)

        policy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=memory.actions,
                                                                     logits=logits)

        advantage = tf.reshape(advantage, policy_loss.shape)

        policy_loss = tf.multiply(policy_loss, tf.stop_gradient(advantage))
        policy_loss -= 0.01 * entropy
        value_loss = tf.reshape(value_loss, policy_loss.shape)
        total_loss = tf.reduce_mean((0.5 * value_loss + policy_loss))
        return total_loss
