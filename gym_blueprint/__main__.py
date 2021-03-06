import gym
import gym_battleship


if __name__ == '__main__':
    env = gym.make('battleship-v0')
    env.reset()
    for _ in range(1000):
        env.render()
        env.step(env.action_space.sample())  # take a random action
    env.close()
