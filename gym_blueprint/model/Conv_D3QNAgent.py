import os
import random
import gym
import cv2
import numpy as np

from .D3QNAgent import D3QNAgent
from .basic_model import CNNModel
from .replay_memory.memory import Memory


class Conv_D3QNAgent(D3QNAgent):
    def __init__(self, game_name, state_size, action_size, epsilon_decay=0.99, memory_size=1000000, gamma=0.99,
                 tau=.001, batch_size=32, dueling=True, ddqn=True, USE_PER=True):

        self.ROWS = 160
        self.COLS = 240
        self.REM_STEP = 4

        self.image_memory = np.zeros((self.REM_STEP, self.ROWS, self.COLS))
        self.state_size = (self.REM_STEP, self.ROWS, self.COLS)
        self.Save_Path = 'Models'
        if not os.path.exists(self.Save_Path):
            os.makedirs(self.Save_Path)
        self.scores, self.episodes, self.average = [], [], []

        super().__init__(game_name=game_name, state_size=self.state_size, action_size=action_size,
                         epsilon_decay=epsilon_decay, memory_size=memory_size, gamma=gamma, tau=tau,
                         batch_size=batch_size, dueling=dueling, ddqn=ddqn, USE_PER=USE_PER)

        # after some time interval update the target model to be same with model

    def _init_models(self, state_size, action_size, learning_rate, dueling):
        return CNNModel(state_size, action_size, learning_rate, dueling)

    def imshow(self, image, rem_step=0):
        cv2.imshow("cartpole" + str(rem_step), image[rem_step, ...])
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            return

    def GetImage(self, img):
        # img = self.env.render(mode='rgb_array')

        img_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_rgb_resized = cv2.resize(img_rgb, (self.COLS, self.ROWS), interpolation=cv2.INTER_CUBIC)
        img_rgb_resized[img_rgb_resized < 255] = 0
        img_rgb_resized = img_rgb_resized / 255

        self.image_memory = np.roll(self.image_memory, 1, axis=0)
        self.image_memory[0, :, :] = img_rgb_resized

        # self.imshow(self.image_memory,0)

        return np.expand_dims(self.image_memory, axis=0)

    def reset_env(self, img):
        for _ in range(self.REM_STEP):
            state = self.GetImage(img)
        return state

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        next_state = self.GetImage(self.env.render(mode='rgb_array'))
        return next_state, reward, done, info

    def act(self, state, epsilon):
        if np.random.random() <= epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.model.predict(state))

    def run(self, trials, plotname="CNN_D3QNAgent"):
        for episode in range(trials):
            list_reward = []
            self.env.reset()
            img = self.env.render(mode='rgb_array')
            state = self.reset_env(img)
            while True:
                action = self.act(state, self.epsilon)
                next_state, reward, done, _ = self.step(action)
                list_reward.append(reward)
                self.memorize(state, action, reward, next_state, done)
                state = next_state
                if done:
                    total_reward = np.sum(list_reward)
                    print("episode: {}/{}, e: {:.2}, total reward: {}"
                          .format(episode, trials, self.epsilon, total_reward))
                    self.plotModel(total_reward, episode, plotname)
                    break
                self.replay(self.batch_size)
            self.update_epsilon()

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

        state = np.zeros((self.batch_size,) + self.state_size)
        next_state = np.zeros((self.batch_size,) + self.state_size)
        action, reward, done = [], [], []

        # do this before prediction
        # for speedup, this could be done on the tensor level
        # but easier to understand using a loop
        for i, elem in enumerate(minibatch):
            state[i] = elem[0]
            action.append(elem[1])
            reward.append(elem[2])
            next_state[i] = elem[3]
            done.append(elem[4])

        # do batch prediction to save speed
        # predict Q-values for starting state using the main network
        target = self.model.predict(state)
        target_old = np.array(target)
        # predict best action in ending state using the main network
        target_next = self.model.predict(next_state)
        # predict Q-values for ending state using the target network
        target_val = self.target_model.predict(next_state)

        for i in range(len(minibatch)):
            # correction on the Q value for the action used
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                # the key point of Double DQN
                # selection of action is from model
                # update is from target model
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

            if self.USE_PER:
                absolute_errors = np.abs(target_old[i] - target[i])
                # Update priority
                self.MEMORY.batch_update(tree_idx, absolute_errors)

        # Train the Neural Network with batches
        self.model.fit(state, target, batch_size=self.batch_size, verbose=0)
        self.update_target_model(self.tau)

    def test(self, total_episode, render="True", name="test_plot"):
        self.scores, self.episodes, self.average = [], [], []
        self.epsilon = 0
        self.env = gym.make(self.game_name)
        mem = Memory()
        for episode in range(total_episode):
            self.env.reset()
            mem.clear()
            ep_reward = 0.

            done = False
            while not done:  # and ep_steps<500:
                img = self.env.render(mode='rgb_array')
                current_state = self.reset_env(img)
                action = self.act(current_state, 0)
                new_state, reward, done, _ = self.env.step(action)

                if render:
                    self.env.render()
                ep_reward += reward
                mem.store(current_state, action, reward)
                if done:  # done and print information
                    self.plotModel(score=ep_reward, episode=episode, name=name)
                    break
