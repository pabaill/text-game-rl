"""
Simple testing script for Jericho library zork game.
"""

from jericho import *

# Create the environment, optionally specifying a random seed
env = FrotzEnv("../jericho/z-machine-games-master/jericho-game-suite/zork1.z5")
state, info = env.reset()
done = False
while not done:
    print('Total Score', info['score'], 'Moves', info['moves'])
    print(state)
    act = input("Action: ")
    state, reward, done, info = env.step(act)
    if reward > 0:
        print(f"Reward! {reward}")
print(state)
print('THE END')

# Walkthrough method
walkthrough = env.get_walkthrough()
for act in walkthrough:
    # Take an action in the environment using the step fuction.
    # The resulting text-observation, reward, and game-over indicator is returned.
    # move = input("Input: ")
    move = input("Press ENTER to advance")
    observation, reward, done, info = env.step(act)
    print(f"Action: {act}")
    print(observation)
    print(f"Reward: {reward}")
    print(f"Next valid actions: {env.get_valid_actions()}")
    # Total score and move-count are returned in the info dictionary
    print('Total Score', info['score'], 'Moves', info['moves'])
print('Scored', info['score'], 'out of', env.get_max_score())