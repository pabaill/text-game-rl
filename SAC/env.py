import gym
from jericho import *

class TextAdventureEnv(gym.Env):
    def __init__(self, game_path):
        self.game = FrotzEnv(game_path)
        self.valid_actions = list(self.game.get_dictionary())

    def reset(self):
        self.state = self.game.reset()
        return self.state

    def step(self, action: str):
        next_state, reward, done, info = self.game.step(action)
        return next_state, reward, done, info