import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym.envs.toy_text import discrete
import numpy as np


class BattleshipEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.shape = (6, 6)
        self.cases = 3
        self.hit = 0

        # state
        self._board = np.zeros(self.shape, dtype=np.bool)
        for i in range(3):
            self._board[2,1+i] = True

        self.observed_board = np.zeros(self.shape)

        # Example when using discrete actions:
        self.action_space = spaces.Tuple((spaces.Discrete(self.shape[0]), spaces.Discrete(self.shape[0])))

        self.state = (self.observed_board, self._board)

        #super(BattleshipEnv, self).__init__()

    def step(self, action):
        action_x = action//6
        action_y = action - 6 * (action//6)
        action = (action_x, action_y)

        state = self.state
        if self.observed_board[action]!=0:
            print("Nimp")
            return np.array(self.state), -1000, True, {}

        else:
            if self._board[action] == True:
                self.observed_board[action] = 2
                #self.state = (self.observed_board, self._board)
                self.hit += 1
                if self.hit == self.cases:
                    print("Victory")
                    return np.array(self.state), 10, True, {}
                else:
                    return np.array(self.state), 1, False, {}
            else:
                self.observed_board[action] = -1
                return np.array(self.state), -1, False, {}

    def render(self):
        for i in range(3):
            for j in range(3):
                self.state[i][j] = "-"
        self.counter = 0
        self.done = 0
        self.add = [0, 0]
        self.reward = 0
        return self.state

    def reset(self):
        self.shape = (6, 6)
        self.cases = 3
        self.hit = 0

        # state
        self._board = np.zeros(self.shape, dtype=np.bool)
        for i in range(3):
            self._board[2,1+i] = True

        self.observed_board = np.zeros(self.shape)
        self.state = (self.observed_board, self._board)

        return np.array(self.observed_board)