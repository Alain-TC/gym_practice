from collections import deque
import random
import numpy as np
from keras.models import Model
from keras.layers import Dense, Input
from keras.layers.merge import Add
from keras.optimizers import Adam
import tensorflow as tf
from .basic_agent import Agent
import keras.backend as K

'''
Pathwise Derivative Policy Gradient Methods
Actor Critic with target networks for continuous action_space
'''


class ActorCritic(Agent):
    def __init__(self, state_size, action_size, epsilon_decay=0.99, memory_size=1000, gamma=0.99, tau=.001):
        super().__init__(state_size, action_size, epsilon_decay, memory_size, gamma)

        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.tau = tau
        self.memory = deque(maxlen=memory_size)
        sess = tf.Session()
        K.set_session(sess)
        self.sess = sess

        # ===================================================================== #
        #                              Actor Model                              #
        # ===================================================================== #
        self.actor_state_input, self.actor_model = self.create_actor_model(self.state_size, self.action_size[0],
                                                                           self.learning_rate)
        _, self.target_actor_model = self.create_actor_model(self.state_size, self.action_size[0], self.learning_rate)

        self.actor_target_weights = self.target_actor_model.get_weights()

        self.actor_critic_grad = tf.placeholder(tf.float32, [None, self.action_size[0]])

        actor_model_weights = self.actor_model.trainable_weights
        self.actor_grads = tf.gradients(self.actor_model.output,
                                        actor_model_weights, -self.actor_critic_grad)  # dC/dA (from actor)
        grads = zip(self.actor_grads, actor_model_weights)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)
        #Adam(loss)
        ##model.compile(loss=_huber_loss, optimizer=Adam(lr=learning_rate))

        # ===================================================================== #
        #                              Critic Model                             #
        # ===================================================================== #

        self.critic_state_input, self.critic_action_input, \
        self.critic_model = self.create_critic_model(self.state_size, self.action_size, self.learning_rate)
        _, _, self.target_critic_model = self.create_critic_model(self.state_size, self.action_size, self.learning_rate)

        self.critic_grads = tf.gradients(self.critic_model.output,
                                         self.critic_action_input)

        # Initialize for later gradient calculations
        self.sess.run(tf.initialize_all_variables())

    def create_actor_model(self, input_shape, output_shape, learning_rate):
        state_input = Input(shape=input_shape)
        h1 = Dense(24, activation='relu')(state_input)
        h2 = Dense(48, activation='relu')(h1)
        h3 = Dense(24, activation='relu')(h2)
        output = Dense(output_shape, activation='relu')(h3)

        model = Model(input=state_input, output=output)
        adam = Adam(lr=learning_rate)
        model.compile(loss="mse", optimizer=adam)
        return state_input, model

    def create_critic_model(self, input_shape, action_shape, learning_rate):
        state_input = Input(shape=input_shape)
        state_h1 = Dense(24, activation='relu')(state_input)
        state_h2 = Dense(48)(state_h1)

        action_input = Input(shape=action_shape)
        action_h1 = Dense(48)(action_input)

        merged = Add()([state_h2, action_h1])
        merged_h1 = Dense(24, activation='relu')(merged)
        output = Dense(1, activation='relu')(merged_h1)
        model = Model(input=[state_input, action_input],
                      output=output)

        adam = Adam(lr=learning_rate)
        model.compile(loss="mse", optimizer=adam)
        return state_input, action_input, model

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        samples = random.sample(self.memory, batch_size)
        self._train_critic(samples, batch_size)
        self._train_actor(samples, batch_size)
        self.update_target(self.tau)

    def _train_actor(self, minibatch, batch_size):
        state = np.zeros((batch_size, self.state_size[0]))
        next_state = np.zeros((batch_size, self.state_size[0]))
        action, reward, done = [], [], []

        for i, elem in enumerate(minibatch):
            state[i] = elem[0]
            action.append(elem[1])
            reward.append(elem[2])
            next_state[i] = elem[3]
            done.append(elem[4])

        predicted_action = self.actor_model.predict(state)

        grads = self.sess.run(self.critic_grads, feed_dict={
            self.critic_state_input: state,
            self.critic_action_input: predicted_action
        })[0]

        self.sess.run(self.optimize, feed_dict={
            self.actor_state_input: state,
            self.actor_critic_grad: grads
        })

    def _train_critic(self, minibatch, batch_size):
        state = np.zeros((batch_size, self.state_size[0]))
        next_state = np.zeros((batch_size, self.state_size[0]))
        action, reward, done = [], [], []

        for i, elem in enumerate(minibatch):
            state[i] = elem[0]
            action.append(elem[1])
            reward.append(elem[2])
            next_state[i] = elem[3]
            done.append(elem[4])

        # do batch prediction to save speed
        target_action = self.target_actor_model.predict(next_state)
        future_reward = self.target_critic_model.predict([next_state, target_action])
        reward += self.gamma * future_reward

        self.critic_model.fit([state, np.array([x[0][0] for x in action])], reward, verbose=0)

    def _update_actor_target(self, tau=1):
        actor_model_weights = self.actor_model.get_weights()
        actor_target_weights = self.target_actor_model.get_weights()

        # copy weights from model to target_model
        new_weights = []
        for i, actor_target_weight in enumerate(actor_target_weights):
            new_weights.append(tau * actor_model_weights[i] + (1 - tau) * actor_target_weight)
        self.actor_target_weights = new_weights
        self.target_actor_model.set_weights(new_weights)

    def _update_critic_target(self, tau):
        critic_model_weights = self.critic_model.get_weights()
        critic_target_weights = self.target_critic_model.get_weights()

        new_weights = []
        for i, critic_target_weight in enumerate(critic_target_weights):
            new_weights.append(tau * critic_model_weights[i] + (1 - tau) * critic_target_weight)
        self.target_critic_model.set_weights(new_weights)

    def update_target(self, tau):
        self._update_actor_target(tau)
        self._update_critic_target(tau)

    def act(self, state, env):
        if np.random.random() < self.epsilon:
            return env.action_space.sample()
        return self.actor_model.predict(state)
