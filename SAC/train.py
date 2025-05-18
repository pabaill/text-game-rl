from env import TextAdventureEnv
from llama import LLaMAWrapper
from ac import Actor, Critic
from replay import ReplayBuffer
from copy import deepcopy
import torch
from torch import nn

import wandb


def train(game_path, _lambda=0.1, lra=1e-4, lrc=1e-4, batch_size=32, episodes=1000, gamma=0.9, action_dim=512, state_dim=4096):
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
    target_critic = deepcopy(critic)

    replay_buffer = ReplayBuffer()

    optimizer_actor = torch.optim.Adam(actor.parameters(), lr=lra)
    optimizer_critic = torch.optim.Adam(critic.parameters(), lr=lrc)

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

                with torch.no_grad():
                    a_next = actor(s_next)
                    q_target = r + gamma * (1 - d) * target_critic(s_next, a_next)

                q_current = critic(s, a)
                critic_loss = nn.MSELoss()(q_current, q_target)
                optimizer_critic.zero_grad()
                critic_loss.backward()
                optimizer_critic.step()

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
            "reward": episode_reward,  # You can log average reward per episode or any other metrics
            "raw_reward": episode_raw_reward
        })
        if episode % 100 == 0:
            # Save the model after training
            torch.save(actor.state_dict(), f"actor_model_ckpt_{episode}.pth")
            torch.save(critic.state_dict(), f"critic_model_ckpt_{episode}.pth")
            wandb.save(f"actor_model_ckpt_{episode}.pth")
            wandb.save(f"critic_model_ckpt_{episode}.pth")

if __name__ == '__main__':
    game = input('Choose game: ')
    game_path = f"../jericho/z-machine-games-master/jericho-game-suite/{game}.z5"
    train(game_path)