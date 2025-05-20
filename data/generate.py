from jericho import *
from tqdm import tqdm
import csv
import random

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

def save_to_csv(data, filename="training_data.csv"):
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
    env = FrotzEnv(f"../jericho/z-machine-games-master/jericho-game-suite/{gamename}.z5")
    data = []
    for i in range(n_walkthroughs):
        prev_observation, info = env.reset()
        walkthrough = env.get_walkthrough()
        for action in tqdm(walkthrough, unit="action"):
            chosen_action = action
            # Add some randomness to walkthrough data
            if random.random() < p_rand:
                chosen_action = random.choice(env.get_valid_actions())

            # Take an action in the environment
            observation, reward, done, info = env.step(chosen_action)
            
            # Capture the current state, action, reward, next state, and done flag
            data.append((prev_observation, chosen_action, reward, observation, done))
            prev_observation = observation
            if done:
                prev_observation, info = env.reset()
    save_to_csv(data, filename=f"training_data_{gamename}.csv")

if __name__ == '__main__':
    gamename = input("Game name: ")
    generate_dataset(gamename)