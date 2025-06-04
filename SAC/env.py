"""
A custom OpenAI Gym environment for text adventure games using the Jericho library.
Provides an interface for interacting with text adventure games like Zork,
with support for state management, action validation, and walkthrough-based exploration.
"""

import gym
from jericho import *
import random


class TextAdventureEnv(gym.Env):
    """
    Has three available variables:
    game: the game object itself from jericho
    valid_actions: list of strings of valid game actions
    game_states: list of possible game states in walkthrough in tuple (state_text, state_embedding)
    """
    def __init__(self, game_path):
        self.game = FrotzEnv(game_path)

        # tuples of (state_text, state_embedding)
        game_states = []
        env = FrotzEnv(game_path)
     
        # build a lit of game states from walkthrough
        walkthrough = env.get_walkthrough()
        state_text, _ = env.reset()
        for action in walkthrough:
            game_states.append((state_text, env.get_state()))
            state_text, _, _, _ = env.step(action)
        self.game_states = game_states

    def reset(self, random_reset=True):
        # reset to a random state
        if random_reset:
            state = random.choice(self.game_states)
            self.game.set_state(state[1])
            return state[0]
        
        # reset to the first state
        else:
            state, _ = self.game.reset()
            return state
    
    def reset_to_state(self, start_idx):
        # reset to a specific state
        state_text, state_embed = self.game_states[start_idx]
        self.game.set_state(state_embed)
        return state_text

    def get_valid_actions(self):
        return self.game.get_valid_actions()

    def step(self, action: str):
        next_state, reward, done, info = self.game.step(action)
        state_to_rewind_to = self.game.get_state()

        # Look around to get more environment info then rewind
        look, _, _, _ = self.game.step('look')
        self.game.set_state(state_to_rewind_to)
        inventory = ",".join([item.name for item in self.game.get_inventory()])
        next_state = f"Look: {look} Holding: {inventory}. State: {next_state}"
        return next_state, reward, done, info
    
    def get_dictionary(self):
        return self.game.get_dictionary()
