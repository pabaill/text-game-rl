import gym
from jericho import *
import random

class TextAdventureEnv(gym.Env):
    def __init__(self, game_path):
        self.game = FrotzEnv(game_path)
        # Tuples of (state_text, state_embedding)
        game_states = []
        env = FrotzEnv(game_path)
        walkthrough = env.get_walkthrough()
        state_text, _ = env.reset()
        for action in walkthrough:
            game_states.append((state_text, env.get_state()))
            state_text, _, _, _ = env.step(action)
        self.game_states = game_states

    def reset(self):
        state = random.choice(self.game_states)
        self.game.set_state(state[1])
        return state[0]
    
    def get_valid_actions(self):
        return self.game.get_valid_actions()

    def step(self, action: str):
        next_state, reward, done, info = self.game.step(action)
        inventory = ",".join([item.name for item in self.game.get_inventory()])
        next_state = f"Holding: {inventory}. State: {next_state}"
        return next_state, reward, done, info