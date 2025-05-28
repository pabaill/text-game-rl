from jericho import *

# Create the environment, optionally specifying a random seed
env = FrotzEnv("../jericho/z-machine-games-master/jericho-game-suite/yomomma.z8")
initial_observation, info = env.reset()
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