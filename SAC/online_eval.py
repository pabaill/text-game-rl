"""
Main online evaluation script for trained SAC actor model.
"""

import torch
from llama import LLaMAWrapper
from ac import Actor
from env import TextAdventureEnv
from online_train import generate_embedding_action_dict, decode_action
import argparse
import pandas as pd


def eval(game_path, actor_ckpt_path, state_dim=3072, action_dim=3072, max_ep_len=200, output_file="results.csv"):
    """
    Evaluation function for trained SAC actor model.
    Args:
        game_path (str): Path to the game to evaluate on (usually zork)
        actor_ckpt_path (str): Path to the actor checkpoint
        state_dim (int): State embedding dimension (3072)
        action_dim (int): Action embedding dimension (3072)
        max_ep_len (int): Maximum episode length (1000 used in paper)
        output_file (str): Path to save results (results.csv)

    Returns:
        saved results.csv containing game history file, and prints summary stats
    """
    # initialize environment, LLM embeddor, actor
    env = TextAdventureEnv(game_path)
    llama = LLaMAWrapper()
    actor = Actor(state_dim=state_dim, action_dim=action_dim)
    actor.load_state_dict(torch.load(actor_ckpt_path, map_location=torch.device('cpu')))
    actor.eval()

    # initialize game data and metadata 
    state_text, info = env.game.reset()
    direction_count, meta_count, interaction_count = 0, 0, 0
    game_dict = env.get_dictionary()
    nav_words = [item.word for item in game_dict if item.is_dir]
    meta_words = [item.word for item in game_dict if item.is_meta]

    # track progress through episodes
    results = []
    done = False
    ep_rem = max_ep_len

    # maintain embedding_to_action dict to build up over time
    embedding_to_action = {}

    # progress through episode by taking actions in environment and tracking results
    while not done and ep_rem > 0:
        valid_actions = env.get_valid_actions()

        # generate embedding_to_action dict (first pass essentially initializes it, subsequent passes update it)
        embedding_to_action = generate_embedding_action_dict(embedding_to_action, valid_actions, llama)

        # encode current state
        state_embedding = llama.encode_text(state_text)

        # convert from [1, 3072] to [3072] shape for decoder
        action_embedding = actor(state_embedding).squeeze(0)

        # decode action to achieve text 
        action_text = decode_action(action_embedding, embedding_to_action, is_training=False)

        # take action in environment
        next_state_text, reward, done, info = env.step(action_text)

        # track results
        results.append({
            "state": state_text,
            "action": action_text,
            "next_state": next_state_text,
            "reward": reward,
            "current_score": info['score']
        })

        # track data about actions taken
        if action_text in nav_words:
            direction_count += 1
        if action_text in meta_words:
            meta_count += 1
        else:
            interaction_count += 1

        # increcment
        state_text = next_state_text
        ep_rem -= 1

    # save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)

    # summary stats
    total_reward = sum([result['reward'] for result in results])
    avg_reward = total_reward / len(results)
    print(f"Average Reward: {avg_reward}")

    final_score = info['score']
    print(f"Final Score: {final_score}")

    total_actions = direction_count + meta_count + interaction_count
    episode_length = max_ep_len - ep_rem
    print(f"Episode Length: {episode_length}")
    print(f"Total actions: {total_actions} | Navigation: {direction_count / total_actions} | Meta: {meta_count / total_actions} | Interaction: {interaction_count / total_actions}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train RL agent from CSV dataset.")
    parser.add_argument('--game_path', type=str, required=True, help='Path to the game to learn')
    parser.add_argument('--actor_ckpt_path', type=str, required=True, help="path to actor checkpoint")
    parser.add_argument('--output_file_path', type=str, required=True, help="path to output file")
    parser.add_argument('--max_ep_len', type=int, required=False, default=200, help="max episode length")
    args = parser.parse_args()
    eval(
        args.game_path,
        args.actor_ckpt_path,
        max_ep_len=args.max_ep_len,
        output_file=args.output_file_path
    )