from collections import deque
import random
import tensorflow as tf
from keras.models import Model
from keras.layers.merge import Add
from keras.optimizers import Adam
from .basic_agent import Agent
from .replay_memory.memory import Memory
from keras.layers import Dense, Input, Lambda
from tensorflow.python import keras
from tensorflow.python.keras import layers
import numpy as np
tf.enable_eager_execution()

'''
Actor Critic for continuous action_space
Critic computed through Value Function Estimation -> Advantage Actor Critic
'''


class ActorCriticModel(keras.Model):
    def __init__(self, state_size, action_size):
        print(state_size, action_size)
        super(ActorCriticModel, self).__init__()
        self.state_size = state_size
        self.action_size = action_size[0]
        self.dense_actor_1 = layers.Dense(8, activation='relu')
        self.dense_actor_2 = layers.Dense(8, activation='relu')
        self.mu_dense = layers.Dense(self.action_size)
        self.sigma_dense = layers.Dense(self.action_size)
        self.dense_critic_1 = layers.Dense(8, activation='relu')
        self.dense_critic_2 = layers.Dense(8, activation='relu')
        self.values = layers.Dense(1)

    def call(self, inputs):
        # Forward pass
        x_1 = self.dense_actor_1(inputs)
        x_2 = self.dense_actor_2(x_1)
        mu = self.mu_dense(x_2)
        sigma = self.sigma_dense(x_2)
        sigma = tf.nn.softplus(sigma) + 1e-5
        v1 = self.dense_critic_1(inputs)
        v2 = self.dense_critic_2(v1)
        values = self.values(v2)
        norm_dist = tf.contrib.distributions.Normal(mu, sigma)
        action_tf_var = tf.squeeze(norm_dist.sample(1), axis=0)
        action_tf_var = tf.clip_by_value(action_tf_var,
                                         -2,
                                         2)


        return (action_tf_var, norm_dist), values


class ActorCritic(Agent):
    def __init__(self, env, state_size, action_size, epsilon_decay=0.99, memory_size=1000, gamma=0.99, tau=.001):
        super().__init__(env, state_size, action_size, epsilon_decay, memory_size, gamma)

        self.env = env
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.tau = tau
        self.lr = .001
        self.memory = deque(maxlen=memory_size)

        #        self.actor_critic_grad = tf.placeholder(tf.float32, [None, self.action_size[0]])
        #
        #        actor_model_weights = self.actor_model.trainable_weights
        #        self.actor_grads = tf.gradients(self.actor_model.output,
        #                                        actor_model_weights, -self.actor_critic_grad)  # dC/dA (from actor)
        #        grads = zip(self.actor_grads, actor_model_weights)
        #        self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)
        # Adam(loss)
        ##model.compile(loss=_huber_loss, optimizer=Adam(lr=learning_rate))



        # ===================================================================== #
        #                              FROM A3C                                 #
        # ===================================================================== #



        self.opt = tf.train.AdamOptimizer(self.lr, use_locking=True)
        print(self.state_size, self.action_size)

        self.model = ActorCriticModel(self.state_size, self.action_size)  # global network

        self.model(tf.convert_to_tensor(np.random.random((1, self.state_size)), dtype=tf.float32))
        print("summary")
        self.model.summary()

    def create_actor_model(self, input_shape, output_shape, learning_rate):
        # state = Input(batch_shape=(None, self.state_size))
        state_input = Input(shape=input_shape)
        h1 = Dense(24, activation='relu')(state_input)
        h2 = Dense(48, activation='relu')(h1)
        h3 = Dense(24, activation='relu')(h2)
        # output = Dense(output_shape, activation='relu')(h3)

        # model = Model(input=state_input, output=output)
        # adam = Adam(lr=learning_rate)
        # model.compile(loss="mse", optimizer=adam)


        mu_0 = Dense(output_shape, activation='tanh', kernel_initializer='he_uniform')(h3)
        sigma_0 = Dense(output_shape, activation='softplus', kernel_initializer='he_uniform')(h3)

        mu = Lambda(lambda x: x * 2)(mu_0)
        sigma = Lambda(lambda x: x + 0.0001)(sigma_0)

        model = Model(inputs=state_input, outputs=(mu, sigma))

        return state_input, model

    def create_value_model(self, input_shape, output_shape, learning_rate):
        state_input = Input(shape=input_shape)
        h1 = Dense(24, activation='relu')(state_input)
        h2 = Dense(48, activation='relu')(h1)
        h3 = Dense(24, activation='relu')(h2)
        output = Dense(output_shape, activation='relu')(h3)

        model = Model(input=state_input, output=output)
        # adam = Adam(lr=learning_rate)
        # model.compile(loss="mse", optimizer=adam)
        return state_input, model

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        samples = random.sample(self.memory, batch_size)
        self._train_critic(samples, batch_size)
        self._train_actor(samples, batch_size)
        self.update_target(self.tau)

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

    def compute_loss(self, done, new_state, memory, gamma=0.99):
        if done:
            reward_sum = 0.  # terminal
        else:
            reward_sum = self.model(tf.convert_to_tensor(new_state[None, :], dtype=tf.float32))[-1].numpy()[0]

        # Get discounted rewards
        discounted_rewards = []
        for reward in memory.rewards[::-1]:  # reverse buffer r
            reward_sum = reward + gamma * reward_sum
            discounted_rewards.append(reward_sum)
        discounted_rewards.reverse()

        a, values = self.model(tf.convert_to_tensor(np.vstack(memory.states), dtype=tf.float32))
        mu, sigma = a
        # self.normal_dist = tf.contrib.distributions.Normal(mu, sigma)
        # self.action = tf.clip_by_value(self.normal_dist.sample(1), env.action_space.low[0], env.action_space.high[0])

        # pdf = 1. / K.sqrt(2. * np.pi * sigmas) * K.exp(-K.square(action - mu) / (2. * sigma_sq))

        # policy_loss = -tf.log(norm_dist.prob(action_tf_var) + 1e-5)# * delta_placeholder

        # values = self.value_model(tf.convert_to_tensor(np.vstack(memory.states), dtype=tf.float32))
        # Get our advantages
        advantage = tf.convert_to_tensor(np.array(discounted_rewards)[:, None], dtype=tf.float32) - values
        # Value loss
        value_loss = advantage ** 2

        norm_dist = tf.contrib.distributions.Normal(mu, sigma)
        action_tf_var = tf.squeeze(norm_dist.sample(1), axis=0)
        action_tf_var = tf.clip_by_value(action_tf_var,
                                         self.env.action_space.low[0],
                                         self.env.action_space.high[0])
        # define actor (policy) loss function
        policy_loss = -tf.log(norm_dist.prob(action_tf_var)
                              + 1e-5)  # * delta_placeholder
        # training_op_actor = tf.train.AdamOptimizer(.001, name='actor_optimizer').minimize(loss_actor)


        # Calculate our policy loss
        # policy = tf.nn.softmax(logits)

        # policy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=memory.actions,
        #                                                             logits=logits)
        policy_loss *= tf.stop_gradient(advantage)
        total_loss = tf.reduce_mean((0.5 * value_loss + policy_loss))
        return total_loss

    def run(self):
        update_freq = 20
        max_episodes = 1000
        episode = 0
        self.lr = .001
        total_step = 1
        mem = Memory()

        while episode < max_episodes:
            current_state = self.env.reset()
            mem.clear()
            ep_reward = 0.
            ep_steps = 0
            self.ep_loss = 0

            time_count = 0
            done = False
            while not done:
                a, b = self.model(tf.convert_to_tensor(current_state[None, :], dtype=tf.float32))
                mu, sigma = a
                # print(a)
                epsilon = np.random.randn(self.action_size[0])
                mu, sigma = mu.numpy()[0], sigma.numpy()[0]
                action = mu + np.sqrt(sigma + .0001) * epsilon
                action = np.clip(action, -2, 2)

                new_state, reward, done, _ = self.env.step(action)
                if done:
                    reward = -1
                ep_reward += reward
                # print(reward)
                mem.store(current_state, action, reward)
                if done:
                    print(ep_reward)
                if time_count == update_freq or done:
                    # Calculate gradient wrt to local model. We do so by tracking the
                    # variables involved in computing the loss by using tf.GradientTape
                    with tf.GradientTape() as tape:
                        total_loss = self.compute_loss(done, new_state, mem, self.gamma)
                    self.ep_loss += total_loss
                    # Calculate local gradients
                    grads = tape.gradient(total_loss, self.model.trainable_weights)
                    # Push local gradients to global model
                    self.opt.apply_gradients(zip(grads,
                                                 self.model.trainable_weights))
                    mem.clear()
                    time_count = 0

                    if done:  # done and print information
                        episode += 1
                        self.plotModel(score=ep_reward, episode=episode, name="actor_critic")

                ep_steps += 1

                time_count += 1
                current_state = new_state
                total_step += 1
