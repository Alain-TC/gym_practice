from .basic_model import SmallModel
from .DDQNAgent import DDQNAgent


class D3QNAgent(DDQNAgent):
    def __init__(self, state_size, action_size, epsilon_decay=0.99, memory_size=1000000, gamma=0.99, tau=.001):
        super().__init__(state_size, action_size, epsilon_decay, memory_size, gamma, tau)

        self.model = SmallModel(self.state_size, self.action_size, self.learning_rate, True)
        self.target_model = SmallModel(self.state_size, self.action_size, self.learning_rate, True)
        self.update_target_model(1)
