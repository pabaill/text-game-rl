from jericho import *
# Create the environment, optionally specifying a random seed
env = FrotzEnv("../jericho/z-machine-games-master/jericho-game-suite/zork1.z5")
initial_observation, info = env.reset()
done = False
while not done:
    # Take an action in the environment using the step fuction.
    # The resulting text-observation, reward, and game-over indicator is returned.
    move = input("Input: ")
    observation, reward, done, info = env.step(move)
    print(observation)
    print(f"Reward: {reward}")
    print(f"Next valid actions: {env.get_valid_actions()}")
    # Total score and move-count are returned in the info dictionary
    print('Total Score', info['score'], 'Moves', info['moves'])
print('Scored', info['score'], 'out of', env.get_max_score())