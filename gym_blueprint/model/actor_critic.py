from collections import deque
import tensorflow as tf
from .basic_agent import Agent
from .replay_memory.memory import Memory
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
        self.dense_actor_1 = layers.Dense(64, activation='relu')
        self.dense_actor_2 = layers.Dense(64, activation='relu')
        self.mu_dense = layers.Dense(self.action_size)
        self.sigma_dense = layers.Dense(self.action_size)
        self.dense_critic_1 = layers.Dense(64, activation='relu')
        self.dense_critic_2 = layers.Dense(64, activation='relu')
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
        return norm_dist, values


class ActorCritic(Agent):
    def __init__(self, env, state_size, action_size, epsilon_decay=0.99, memory_size=1000, gamma=0.99, tau=.001):
        super().__init__(env, state_size, action_size, epsilon_decay, memory_size, gamma)

        self.env = env
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.tau = tau
        self.lr = .0005
        self.memory = deque(maxlen=memory_size)

        self.opt = tf.train.AdamOptimizer(self.lr, use_locking=True)
        self.model = ActorCriticModel(self.state_size, self.action_size)
        self.model(tf.convert_to_tensor(np.random.random((1, self.state_size)), dtype=tf.float32))
        print("summary")
        self.model.summary()

    def compute_loss(self, done, new_state, memory, gamma=0.99):
        if done:
            reward_sum = 0.  # terminal
        else:
            reward_sum = self.model(tf.convert_to_tensor(new_state[None, :], dtype=tf.float32))[-1].numpy()[0]

        # Get discounted rewards
        discounted_rewards = []
        for reward in list(memory.rewards[::-1]):  # reverse buffer r
            reward_sum = reward + gamma * reward_sum
            discounted_rewards.append(reward_sum)
        discounted_rewards.reverse()

        norm_dists, values = self.model(tf.convert_to_tensor(np.vstack(memory.states), dtype=tf.float32))

        logs_probs = norm_dists.log_prob(memory.actions)

        entropy = norm_dists.entropy()

        r = tf.reshape(tf.convert_to_tensor(np.array(discounted_rewards)[:, None], dtype=tf.float32), values.shape)
        # Get our advantages
        advantage = r - values
        self.entropy_beta = .0001

        advantage = tf.reshape(tf.stop_gradient(advantage), logs_probs.shape)

        policy_loss = tf.multiply(-logs_probs, tf.stop_gradient(advantage))  # - entropy * self.entropy_beta
        value_loss = advantage ** 2

        total_loss = tf.reduce_mean((0.5 * value_loss + policy_loss))
        return total_loss

    def run(self):
        update_freq = 10
        max_episodes = 100
        episode = 0
        self.lr = .0005
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
                time_count += 1
                norm_dist, value = self.model(tf.convert_to_tensor(current_state[None, :], dtype=tf.float32))
                actions = norm_dist.sample().numpy()
                action = np.clip(actions, self.env.action_space.low.min(), self.env.action_space.high.max())[0]

                new_state, reward, done, _ = self.env.step(action)
                if done:
                    reward = -1
                ep_reward += reward
                mem.store(current_state, action, reward)
                if done:
                    print(time_count)
                if time_count == update_freq or done:
                    # Calculate gradient wrt to local model. We do so by tracking the
                    # variables involved in computing the loss by using tf.GradientTape
                    with tf.GradientTape(persistent=False) as tape:
                        total_loss = self.compute_loss(done, new_state, mem, self.gamma)

                    # Calculate local gradients
                    grads = tape.gradient(total_loss, self.model.trainable_weights)
                    self.opt.apply_gradients(zip(grads, self.model.trainable_weights))

                    mem.clear()
                    time_count = 0

                    if done:  # done and print information
                        episode += 1
                        self.plotModel(score=ep_reward, episode=episode, name="actor_critic")

                ep_steps += 1

                time_count += 1
                current_state = new_state
                total_step += 1
