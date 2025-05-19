from env import TextAdventureEnv
from llama import LLaMAWrapper
from ac import Actor, Critic
from replay import ReplayBuffer
from shaper import RewardShaper

import argparse
from copy import deepcopy
import torch
from torch import nn

import wandb


def train(game_path, _lambda=0.1, lra=1e-4, lrc=1e-4, batch_size=32, episodes=1000, gamma=0.9, action_dim=512, state_dim=4096, learn_reward_shaping=False):
    wandb.init(project="text-adventure-rl", entity="pabaill")  # Replace with your W&B username and project name
    wandb.config.update({
        "learning_rate_actor": lra,
        "learning_rate_critic": lrc,
        "batch_size": batch_size,
        "episodes": episodes,
        "gamma": gamma,
        "action_dim": action_dim,
        "state_dim": state_dim,
        "reward_shape_lambda": _lambda
    })
    env = TextAdventureEnv(game_path)
    llama = LLaMAWrapper()
    actor = Actor(state_dim=state_dim, action_dim=action_dim)
    critic = Critic(state_dim=state_dim, action_dim=action_dim)
    reward_shaper = RewardShaper(state_dim)
    target_critic = deepcopy(critic)

    replay_buffer = ReplayBuffer()

    optimizer_actor = torch.optim.Adam(actor.parameters(), lr=lra)
    optimizer_critic = torch.optim.Adam(critic.parameters(), lr=lrc)
    optimizer_shaper = torch.optim.Adam(reward_shaper.parameters(), lr=1e-4)

    # Pre-encode all actions in the action bank
    action_texts = env.valid_actions
    action_embeddings = torch.stack([llama.encode_text(a).squeeze(0) for a in action_texts])

    for episode in range(1000):
        state_text = env.reset()
        episode_loss_actor = 0
        episode_loss_critic = 0
        episode_reward = 0
        done = False

        while not done:
            state_embedding = llama.encode_text(state_text).squeeze(0)
            action_embedding = actor(state_embedding)

            action_text = llama.decode_action(action_embedding, action_embeddings, action_texts)

            next_state_text, reward, done, _ = env.step(action_text)
            next_state_embedding = llama.encode_text(next_state_text).squeeze(0)
            
            if learn_reward_shaping:
                # Use net to learn reward shaping method
                potential_diff = reward_shaper(state_embedding, next_state_embedding).squeeze() # Assuming shaper outputs a scalar
                shaped_reward = reward + _lambda * potential_diff
            else:
                # Compute state value functions
                v_current = critic(state_embedding, action_embedding).mean()
                a_next = actor(next_state_embedding)
                v_next = critic(next_state_embedding, a_next).mean()

                # Shape the reward using the critic as a potential function with discount gamma
                v_current_detach = v_current.detach()
                v_next_detach = v_next.detach()
                shaped_reward = reward + _lambda * (gamma * v_next_detach - v_current_detach)

            # Add the shaped reward to the replay buffer
            replay_buffer.add((state_embedding.detach(), action_embedding.detach(), shaped_reward.detach(), next_state_embedding.detach(), done))

            episode_reward += shaped_reward.item()
            episode_raw_reward += reward

            # Training
            if len(replay_buffer.buffer) > batch_size:
                batch = replay_buffer.sample(batch_size)
                s, a, r, s_next, d = zip(*batch)
                s, a, r, s_next, d = map(torch.stack, (s, a, torch.tensor(r).unsqueeze(1), s_next, torch.tensor(d).unsqueeze(1).float()))

                # Either use learned reward shaping phi OR estimate with V
                if learn_reward_shaping:
                    reward_delta = reward_shaper(s, s_next)
                    shaped_r = r + _lambda * reward_delta.detach()
                else:
                    with torch.no_grad():
                        v_current_batch = critic(s, a).mean(dim=1, keepdim=True)
                        a_next_batch = actor(s_next)
                        v_next_batch = critic(s_next, a_next_batch).mean(dim=1, keepdim=True)
                        shaped_r = r + _lambda * (gamma * v_next_batch - v_current_batch)

                with torch.no_grad():
                    a_next = actor(s_next)
                    q_target = shaped_r + gamma * (1 - d) * target_critic(s_next, a_next)

                q_current = critic(s, a)
                critic_loss = nn.MSELoss()(q_current, q_target)
                optimizer_critic.zero_grad()
                critic_loss.backward()
                optimizer_critic.step()

                if learn_reward_shaping:
                    # Define a target for the reward shaper (TD error)
                    td_error = (r + gamma * (1 - d) * critic(s_next, actor(s_next)).detach() - critic(s, a).detach()).squeeze()
                    shaper_loss = nn.MSELoss()(reward_delta.squeeze(), td_error)
                    optimizer_shaper.zero_grad()
                    shaper_loss.backward()
                    optimizer_shaper.step()

                a_pred = actor(s)
                actor_loss = -critic(s, a_pred).mean()
                optimizer_actor.zero_grad()
                actor_loss.backward()
                optimizer_actor.step()
                episode_loss_actor += actor_loss.item()
                episode_loss_critic += critic_loss.item()

                # Soft update
                for target_param, param in zip(target_critic.parameters(), critic.parameters()):
                    target_param.data.copy_(0.995 * target_param.data + 0.005 * param.data)

            state_text = next_state_text

        wandb.log({
            "episode": episode,
            "loss_actor": episode_loss_actor,
            "loss_critic": episode_loss_critic,
            "reward": episode_reward,
            "raw_reward": episode_raw_reward
        })
        if episode % 100 == 0:
            # Save the model after training
            torch.save(actor.state_dict(), f"actor_model_ckpt_{episode}.pth")
            torch.save(critic.state_dict(), f"critic_model_ckpt_{episode}.pth")
            torch.save(reward_shaper.state_dict(), f"reward_shaper_ckpt_{episode}.pth")
            wandb.save(f"actor_model_ckpt_{episode}.pth")
            wandb.save(f"critic_model_ckpt_{episode}.pth")
            wandb.save(f"reward_shaper_ckpt_{episode}.pth")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train RL agent in text adventure game.")
    parser.add_argument('--game', type=str, required=True, help='Name of the game file (without extension)')
    parser.add_argument('--lambda_', type=float, default=0.1, help='Reward shaping lambda')
    parser.add_argument('--lra', type=float, default=1e-4, help='Learning rate for actor')
    parser.add_argument('--lrc', type=float, default=1e-4, help='Learning rate for critic')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--gamma', type=float, default=0.9, help='Discount factor')
    parser.add_argument('--action_dim', type=int, default=512, help='Action embedding dimension')
    parser.add_argument('--state_dim', type=int, default=4096, help='State embedding dimension')
    parser.add_argument('--learn_reward_shaping', type=bool, default=False, help='Optional learned net for reward shaping')

    args = parser.parse_args()
    game_path = f"../jericho/z-machine-games-master/jericho-game-suite/{args.game}.z5"

    train(
        game_path=game_path,
        _lambda=args.lambda_,
        lra=args.lra,
        lrc=args.lrc,
        batch_size=args.batch_size,
        episodes=args.episodes,
        gamma=args.gamma,
        action_dim=args.action_dim,
        state_dim=args.state_dim,
        learn_reward_shaping=args.learn_reward_shaping
    )