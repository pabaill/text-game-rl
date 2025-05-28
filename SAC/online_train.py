import torch
from torch import nn
from torch.nn.functional import cosine_similarity
from replay import ReplayBuffer
from shaper import RewardShaper
from llama import LLaMAWrapper
from ac import Actor, Critic
from env import TextAdventureEnv

from copy import deepcopy
import wandb
import argparse
from tqdm import tqdm
import random


def generate_embedding_action_dict(valid_actions, llama):
    """
    From List[str] of our valid actions, embed each action and store in a dict of
    {action_embedding : action_text} such that decode action can take the action_embedding
    produced by trained actor model and return a valid action text.
    """
    embedding_to_action = {}
    for action in valid_actions:
        # possible optimization: append .to(torch.float{n}) where n is num bits, could save memory
        action_embedding = llama.encode_text(action).squeeze(0)
        embedding_to_action[action_embedding] = action
    return embedding_to_action


def decode_action(action_embedding, embedding_to_action, is_training=True, k=5):
    """
    Decode the action embedding using nucleus sampling.
    """
    similarities = []
    for embedding, action in embedding_to_action.items():
        # get the cosine similarity betwen predicted action embedding and all valid action embeddings
        similarity = cosine_similarity(action_embedding.unsqueeze(0), embedding.unsqueeze(0)).item()
        similarities.append((similarity, action))

    # sort similarities by actual similarity value and get top k, choose a random one
    similarities.sort(key=lambda x: x[0], reverse=True)
    if is_training:
        selected_action = similarities[0]
    else:
        top_k_actions = [action for _, action in similarities[:k]]
        selected_action = random.choice(top_k_actions)
    return selected_action


# PREVIOUSLY:
# def decode_action(action_embedding, valid_actions, llama):
#     # Choose most similar available action
#     valid_action_embeddings = [llama.encode_text(action).squeeze(0) for action in valid_actions]
#     similarities = [cosine_similarity(action_embedding.unsqueeze(0), valid_embedding.unsqueeze(0)).item() 
#                     for valid_embedding in valid_action_embeddings]
#     best_action_idx = similarities.index(max(similarities))
#     return valid_actions[best_action_idx]

def pretrain_critic(env, critic, target_critic, optimizer_critic, llama, replay_buffer, steps=50, batch_size=32, gamma=0.9, max_ep_len=20):
    valid_actions = env.get_valid_actions()
    
    for step in tqdm(range(steps), desc="Pretraining Critic"):
        state_text = env.reset()
        moves_left = max_ep_len
        done = False

        while not done:
            state_embedding = llama.encode_text(state_text).squeeze(0)

            # Random action
            action_text = random.choice(valid_actions)
            action_embedding = llama.encode_text(action_text).squeeze(0)

            next_state_text, reward, done, _ = env.step(action_text)
            next_state_embedding = llama.encode_text(next_state_text).squeeze(0)

            replay_buffer.add((
                state_embedding.detach(),
                action_embedding.detach(),
                torch.tensor([reward]),
                next_state_embedding.detach(),
                done
            ))

            state_text = next_state_text

            if len(replay_buffer.buffer) > batch_size:
                # Sample batch
                batch = replay_buffer.sample(batch_size)
                s, a, r, s_next, d = zip(*batch)
                s = torch.stack(s)
                a = torch.stack(a)
                r = torch.stack(r)
                s_next = torch.stack(s_next)
                d = torch.tensor(d).unsqueeze(1).float()

                with torch.no_grad():
                    a_next = a  # keep same for bootstrapping — or sample randomly
                    q_target = r + gamma * (1 - d) * target_critic(s_next, a_next)

                q_current = critic(s, a)
                critic_loss = nn.MSELoss()(q_current, q_target)

                optimizer_critic.zero_grad()
                critic_loss.backward()
                optimizer_critic.step()

                # Soft update
                for target_param, param in zip(target_critic.parameters(), critic.parameters()):
                    target_param.data.copy_(0.995 * target_param.data + 0.005 * param.data)
            moves_left -= 1
            if moves_left <= 0 or done:
                break



def train(game_path, max_ep_len=50, _lambda=0.1, lra=1e-4, lrc=1e-4, batch_size=32, episodes=1000, gamma=0.9, action_dim=3072, state_dim=3072, learn_reward_shaping=False, eval_interval=100, train_only=False, wandb_proj=None, wandb_entity=None):
    wandb.init(project=wandb_proj, entity=wandb_entity)
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

    # set seed for reproducibility (mainly for decode_action)
    random.seed(16)

    env = TextAdventureEnv(game_path)

    llama = LLaMAWrapper()
    actor = Actor(state_dim=state_dim, action_dim=action_dim)
    critic = Critic(state_dim=state_dim, action_dim=action_dim)
    reward_shaper = RewardShaper(state_dim)
    target_critic = deepcopy(critic)

    # generate the dict of action_text to action_embedding
    valid_actions = env.get_valid_actions()
    embedding_to_action = generate_embedding_action_dict(valid_actions, llama)

    replay_buffer = ReplayBuffer()

    optimizer_actor = torch.optim.Adam(actor.parameters(), lr=lra)
    optimizer_critic = torch.optim.Adam(critic.parameters(), lr=lrc)
    optimizer_shaper = torch.optim.Adam(reward_shaper.parameters(), lr=1e-4)

    for episode in tqdm(range(episodes), unit="episode"):
        episode_loss_actor = 0
        episode_loss_critic = 0
        episode_reward = 0
        episode_raw_reward = 0

        # Reset environment to a random point in the walkthrough
        state_text = env.reset()
        done = False
        moves_remaining = max_ep_len

        print(f"Episode num {episode}")
        
        while not done:
            prev_state_text = state_text

            prev_state_embedding = llama.encode_text(prev_state_text).squeeze(0)
            action_embedding = actor(prev_state_embedding)
            
            # PREVIOUSLY: action_text = decode_action(action_embedding, env.get_valid_actions(), llama)
            action_text = decode_action(action_embedding, embedding_to_action)

            next_state_text, reward, done, info = env.step(action_text)
            next_state_embedding = llama.encode_text(next_state_text).squeeze(0)

            prev_state_text = next_state_text

            if learn_reward_shaping:
                # Use net to learn reward shaping method
                potential_diff = reward_shaper(prev_state_embedding, next_state_embedding).squeeze()
                shaped_reward = reward + _lambda * potential_diff
            else:
                # Compute state value functions
                v_current = critic(prev_state_embedding, action_embedding).mean()
                a_next = actor(next_state_embedding)
                v_next = critic(next_state_embedding, a_next).mean()

                # Shape the reward using the critic as a potential function
                v_current_detach = v_current.detach()
                v_next_detach = v_next.detach()
                shaped_reward = reward + _lambda * (gamma * v_next_detach - v_current_detach)

            # Add the shaped reward to the replay buffer
            replay_buffer.add((prev_state_embedding.detach(), action_embedding.detach(), shaped_reward.detach(), next_state_embedding.detach(), done))

            episode_reward += shaped_reward.item()
            episode_raw_reward += reward

            # Training
            if len(replay_buffer.buffer) > batch_size:
                batch = replay_buffer.sample(batch_size)
                s, a, r, s_next, d = zip(*batch)
                s = torch.stack(s)
                a = torch.stack(a)
                r = torch.tensor(r).unsqueeze(1)
                s_next = torch.stack(s_next)
                d = torch.tensor(d).unsqueeze(1).float()

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
            moves_remaining -= 1
            if moves_remaining < 0:
                done = True

        wandb.log({
            "episode": episode,
            "loss_actor": episode_loss_actor,
            "loss_critic": episode_loss_critic,
            "reward": episode_reward,
            "raw_reward": episode_raw_reward
        })
        print(f"reward = {reward}")

        if episode % 50 == 0:
            # Save the model after training
            torch.save(actor.state_dict(), f"checkpoints/online/actor_model_ckpt_{episode}.pth")
            torch.save(critic.state_dict(), f"checkpoints/online/critic_model_ckpt_{episode}.pth")
            torch.save(reward_shaper.state_dict(), f"checkpoints/online/reward_shaper_ckpt_{episode}.pth")
            wandb.save(f"actor_model_ckpt_{episode}.pth")
            wandb.save(f"critic_model_ckpt_{episode}.pth")
            wandb.save(f"reward_shaper_ckpt_{episode}.pth")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train RL agent from CSV dataset.")
    parser.add_argument('--game_path', type=str, required=True, help='Path to the game to learn')
    parser.add_argument('--lambda_', type=float, default=0.1, help='Reward shaping lambda')
    parser.add_argument('--lra', type=float, default=1e-4, help='Learning rate for actor')
    parser.add_argument('--lrc', type=float, default=1e-4, help='Learning rate for critic')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--gamma', type=float, default=0.9, help='Discount factor')
    parser.add_argument('--action_dim', type=int, default=3072, help='Action embedding dimension')
    parser.add_argument('--state_dim', type=int, default=3072, help='State embedding dimension')
    parser.add_argument('--learn_reward_shaping', type=bool, default=False, help='Optional learned net for reward shaping')
    parser.add_argument('--train_only', type=bool, default=False, help='No eval or test steps')
    parser.add_argument('--wandb_proj', type=str, default=None, help='wandb project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='wandb entity name')

    args = parser.parse_args()
    train(
        args.game_path,
        _lambda=args.lambda_,
        lra=args.lra,
        lrc=args.lrc,
        batch_size=args.batch_size,
        episodes=args.episodes,
        gamma=args.gamma,
        action_dim=args.action_dim,
        state_dim=args.state_dim,
        learn_reward_shaping=args.learn_reward_shaping,
        train_only=args.train_only,
        wandb_proj=args.wandb_proj,
        wandb_entity=args.wandb_entity
    )