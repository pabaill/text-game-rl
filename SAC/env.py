import gym
from jericho import *
import random
from tqdm import tqdm
from itertools import product


class TextAdventureEnv(gym.Env):
    """
    Has three available variables:
    game: the game object itself from jericho
    valid_actions: list of strings of valid game actions
    game_states: list of possible game states in walkthrough in tuple (state_text, state_embedding)
    """
    def __init__(self, game_path):
        self.game = FrotzEnv(game_path)
        # Tuples of (state_text, state_embedding)
        game_states = []
        env = FrotzEnv(game_path)
        # Get all possible actions
        game_dict = env.get_dictionary()
        noun_list = [item.word for item in game_dict if item.is_noun]
        verb_list = [item.word for item in game_dict if item.is_verb]
        valid_actions = []
        for v, n in tqdm(product(verb_list, noun_list), desc="Generating verb-noun pairs", total=len(verb_list)*len(noun_list)):
            valid_actions.append(f"{v} {n}")
        valid_actions.extend([item.word for item in game_dict if item.is_dir])
        self.valid_actions = valid_actions
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
    
    def reset_to_state(self, start_idx):
        state_text, state_embed = self.game_states[start_idx]
        self.game.set_state(state_embed)
        return state_text

    
    def get_valid_actions(self):
        return self.game.get_valid_actions()

    def step(self, action: str):
        next_state, reward, done, info = self.game.step(action)
        inventory = ",".join([item.name for item in self.game.get_inventory()])
        next_state = f"Holding: {inventory}. State: {next_state}"
        return next_state, reward, done, info