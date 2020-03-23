from keras.models import Sequential
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from keras.layers import Dense
from keras.optimizers import Adam
from .basic_agent import Agent


class DQNAgent(Agent):
    def __init__(self, state_size, action_size, epsilon_decay=0.99, memory_size=1000000, gamma=0.99):
        super().__init__(state_size, action_size, epsilon_decay, memory_size, gamma)