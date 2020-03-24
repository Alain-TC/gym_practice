import random
from collections import deque
import os
import pylab
import numpy as np
from keras.models import model_from_json

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class Agent:
    def __init__(self, env, state_size, action_size, epsilon_decay=0.99, memory_size=1000000, gamma=0.99,
                 batch_size=32):
        self.env = env
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.001
        self.epsilon_decay = epsilon_decay
        self.learning_rate = 0.001
        self.model = None
        self.scores, self.episodes, self.average = [], [], []

    def _init_models(self, state_size, action_size, learning_rate, dueling):
        self.model = None

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.random() <= self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.model.predict(state))

    def run(self, trials, name="default"):
        for episode in range(trials):
            list_reward = []
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            while True:
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                list_reward.append(reward)
                next_state = np.reshape(next_state, [1, self.state_size])
                self.memorize(state, action, reward, next_state, done)
                state = next_state
                if done:
                    total_reward = np.sum(list_reward)
                    print("episode: {}/{}, e: {:.2}, total reward: {}"
                          .format(episode, trials, self.epsilon, total_reward))
                    self.plotModel(total_reward, episode, name)
                    break
                self.replay(self.batch_size)
            self.update_epsilon()

    def replay(self, batch_size):
        self.batch_size = batch_size
        # Randomly sample minibatch from the memory
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

        for i in range(self.batch_size):
            # correction on the Q value for the action used
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                # Standard - DQN
                # DQN chooses the max Q value among next actions
                # selection and evaluation of action is on the target Q Network
                # Q_max = max_a' Q_target(s', a')
                target[i][action[i]] = reward[i] + self.gamma * (np.amax(target_next[i]))

        # Train the Neural Network with batches
        self.model.fit(state, target, batch_size=self.batch_size, verbose=0)

    def load(self, name):
        self.model.load_weights(name)
        print("Loaded model weights")

    def save(self, name):
        self.model.save_weights(name)

    def save_json(self, json_path):
        model_json = self.model.to_json()
        with open(json_path, "w") as json_file:
            json_file.write(model_json)

    def load_model(self, json_path):
        json_file = open(json_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        print("Loaded model architecture")

    pylab.figure(figsize=(18, 9))

    def plotModel(self, score, episode, name):
        self.scores.append(score)
        self.episodes.append(episode)
        self.average.append(sum(self.scores[-50:]) / len(self.scores[-50:]))
        pylab.plot(self.episodes, self.scores, 'b')
        pylab.plot(self.episodes, self.average, 'r')
        pylab.ylabel('Score', fontsize=18)
        pylab.xlabel('Steps', fontsize=18)
        try:
            pylab.savefig("{}.png".format(name))
        except OSError:
            pass
        return str(self.average[-1])[:5]
