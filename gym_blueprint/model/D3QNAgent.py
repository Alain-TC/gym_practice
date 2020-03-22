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
from .DDQNAgent import DDQNAgent


class D3QNAgent(DDQNAgent):
    def __init__(self, state_size, action_size, epsilon_decay=0.99, memory_size=1000000, gamma=0.99, tau=.001):
        super().__init__(state_size, action_size, epsilon_decay, memory_size, gamma, tau)

        self.model = SmallModel(self.state_size, self.action_size, self.learning_rate, False)
        self.target_model = SmallModel(self.state_size, self.action_size, self.learning_rate, False)
        self.update_target_model(1)
