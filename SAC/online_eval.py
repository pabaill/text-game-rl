import torch
from llama import LLaMAWrapper
from ac import Actor, Critic
from env import TextAdventureEnv
from online_train import generate_embedding_action_dict, decode_action
import argparse

import pandas as pd

def eval(game_path, actor_ckpt_path, state_dim=3072, action_dim=3072, max_ep_len=200, output_file="results.csv"):
    env = TextAdventureEnv(game_path)
    llama = LLaMAWrapper()
    actor = Actor(state_dim=state_dim, action_dim=action_dim)
    actor.load_state_dict(torch.load(actor_ckpt_path, map_location=torch.device('cpu')))
    actor.eval()
    
    valid_actions = env.get_valid_actions()
    embedding_to_action = generate_embedding_action_dict(valid_actions, llama)

    state_text, info = env.game.reset()

    results = []
    done = False
    ep_rem = max_ep_len
    while not done and ep_rem > 0:
        state_embedding = llama.encode_text(state_text)
        action_embedding = actor(state_embedding)
        action_text = decode_action(action_embedding, embedding_to_action, is_training=False)
        next_state_text, reward, done, info = env.step(action_text)

        results.append({
            "state": state_text,
            "action": action_text,
            "next_state": next_state_text,
            "reward": reward,
            "current_score": info['score']
        })

        state_text = next_state_text
        ep_rem -= 1

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train RL agent from CSV dataset.")
    parser.add_argument('--game_path', type=str, required=True, help='Path to the game to learn')
    parser.add_argument('--actor_ckpt_path', type-str, required=True, help="path to actor checkpoint")
    parser.add_argument('--output_file_path', type-str, required=True, help="path to output file")
    args = parser.parse_args()
    eval(
        args.game_path,
        args.actor_ckpt_path,
        output_file=args.output_file_path
    )