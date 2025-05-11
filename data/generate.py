import csv
from jericho import *
from tqdm import tqdm

def save_to_csv(data, filename="training_data.csv"):
    # Open the CSV file for writing
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header (column names)
        writer.writerow(["state_embedding", "action_embedding", "reward", "next_state_embedding", "done"])

        # Write the transitions to the file
        for transition in data:
            state, action, reward, next_state, done = transition
            writer.writerow([state, action, reward, next_state, done])

def generate_dataset(gamename):
    env = FrotzEnv(f"../jericho/z-machine-games-master/jericho-game-suite/{gamename}.z5")
    prev_observation, info = env.reset()
    walkthrough = env.get_walkthrough()
    data = []
    for action in tqdm(walkthrough, unit="action"):
        # Take an action in the environment
        observation, reward, done, info = env.step(action)
        
        # Capture the current state, action, reward, next state, and done flag
        data.append((prev_observation, action, reward, observation, done))
        prev_observation = observation
    save_to_csv(data, filename=f"training_data_{gamename}.csv")

if __name__ == '__main__':
    gamename = input("Game name: ")
    generate_dataset(gamename)