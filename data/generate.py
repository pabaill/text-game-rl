from jericho import *
from tqdm import tqdm
import csv
import random

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import time

EVAL = ["yomomma", "gold", "jewel", "lurking", "night"]
TEST = ["zork1", "snacktime"]

def save_to_csv(data, filename="training_data.csv"):
    print(f"Saving {len(data)} training examples to {filename}")
    # Open the CSV file for writing
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header (column names)
        writer.writerow(["state", "action", "reward", "next_state", "done"])

        # Write the transitions to the file
        for transition in data:
            state, action, reward, next_state, done = transition
            writer.writerow([state, action, reward, next_state, done])

def generate_dataset(gamename, n_walkthroughs=5, p_rand=0.1):
    env = FrotzEnv(f"../jericho/z-machine-games-master/jericho-game-suite/{gamename}")

    # Time out after 20 minutes
    timeout = time.time() + (60 * 20)

    for i in tqdm(range(n_walkthroughs)):
        prev_observation, info = env.reset()
        walkthrough = env.get_walkthrough() # returns list of action strings needed to complete the game
        for action in tqdm(walkthrough, unit="action"):
            chosen_action = action
            # Add some randomness to walkthrough data
            took_rand_action = random.random() < p_rand and len(env.get_valid_actions()) > 0
            if took_rand_action:
                chosen_action = random.choice(env.get_valid_actions())
                prev_state = env.get_state()

            # Take an action in the environment
            observation, reward, done, info = env.step(chosen_action)
            
            # Capture the current state, action, reward, next state, and done flag
            data.append((prev_observation, chosen_action, reward, observation, done))

            # If we took a random action, get back on track for the walkthrough
            if took_rand_action:
                env.set_state(prev_state)
                observation, reward, done, info = env.step(action)
            prev_observation = observation
            if done or time.time() > timeout:
                break
        prev_observation, info = env.reset()
    folder = "train"
    if gamename[:-3] in EVAL:
        folder = "eval"
    if gamename[:-3] in TEST:
        folder = "test"
    save_to_csv(data, filename=f"{folder}/training_data_{gamename}_p_rand_{p_rand}.csv")

if __name__ == '__main__':
    gamelist = os.listdir('../jericho/z-machine-games-master/jericho-game-suite')
    p_rand = float(input("p_rand: "))
    existing_games = os.listdir('train')
    games_to_skip = ["905.z5", "acorncourt.z5", "advent.z5", "adventureland.z5", "afflicted.z8", "awaken.z5", "balances.z5", "anchor.z8", "ballyhoo.z3"]
    for gamename in gamelist:
        if gamename not in games_to_skip and f"training_data_{gamename}_p_rand_{p_rand}.csv" not in existing_games:
            print(f"Generating data for : {gamename}")
            generate_dataset(gamename, n_walkthroughs=1, p_rand=p_rand)