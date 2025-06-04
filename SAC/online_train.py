"""
Main online training script for SAC actor model.
Contains logic for:
1. generating embedding action dictionary
2. decoding actions using either greedy decoding or nucleus sampling
3. pretraining the critic model on random actions
4. core training loop for SAC model
"""

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


def generate_embedding_action_dict(embedding_to_action, valid_actions, llama):
    """
    From List[str] of our valid actions, embed each action and store in a dict of
    {action_embedding : action_text} such that decode action can take the action_embedding
    produced by trained actor model and return a valid action text.
    Don't recompute actions that are already in your dictionary!
    """
    existing_action_texts = list(embedding_to_action.values())
    for action in valid_actions:
        if action not in existing_action_texts:
            # possible optimization: append .to(torch.float{n}) where n is num bits, could save memory
            action_embedding = llama.encode_text(action).squeeze(0).detach()
            embedding_to_action[action_embedding] = action
    return embedding_to_action


def decode_action(action_embedding, embedding_to_action, is_training=True, k=5):
    """
    Decode the action embedding using nucleus sampling in evaluation, or greedy decoding (top 1) in training.
    """
    similarities = []
    for embedding, action in embedding_to_action.items():
        # get the cosine similarity betwen predicted action embedding and all valid action embeddings
        similarity = cosine_similarity(action_embedding.unsqueeze(0), embedding.unsqueeze(0)).item()
        similarities.append((similarity, action))

    # sort similarities by actual similarity value
    similarities.sort(key=lambda x: x[0], reverse=True)

    # in training, select top 1 action
    if is_training:
        selected_action = similarities[0][1]

    # in evaluation, do nucleus sampling by choosing 1 of top k actions
    else:
        top_k_actions = [action for _, action in similarities[:k]]
        selected_action = random.choice(top_k_actions)
    return selected_action


def pretrain_critic(env, critic, target_critic, optimizer_critic, llama, replay_buffer, steps=10, batch_size=32, gamma=0.9, max_ep_len=20):
    """
    Pretrain the critic model on random actions.
    """
    # grab all valid actions from environment
    valid_actions = env.get_valid_actions()

    # take 10 steps (by default)
    for step in tqdm(range(steps), desc="Pretraining Critic"):
        # reset environment to start for each episode
        state_text = env.reset()
        moves_left = max_ep_len
        done = False

        # progress through episode by taking random actions in environment
        while not done:
            state_embedding = llama.encode_text(state_text).squeeze(0)

            # take a random action
            action_text = random.choice(valid_actions)
            action_embedding = llama.encode_text(action_text).squeeze(0)

            # step through environment 
            next_state_text, reward, done, _ = env.step(action_text)
            next_state_embedding = llama.encode_text(next_state_text).squeeze(0)

            # build up replay buffer
            replay_buffer.add((
                state_embedding.detach(),
                action_embedding.detach(),
                torch.tensor([reward]),
                next_state_embedding.detach(),
                done
            ))

            # increment
            state_text = next_state_text

            # if we have enough data, sample a batch and train the critic
            if len(replay_buffer.buffer) > batch_size:
                # sample a batch
                batch = replay_buffer.sample(batch_size)
                s, a, r, s_next, d = zip(*batch)
                s = torch.stack(s)
                a = torch.stack(a)
                r = torch.stack(r)
                s_next = torch.stack(s_next)
                d = torch.tensor(d).unsqueeze(1).float()

                # compute target Q-values
                with torch.no_grad():
                    a_next = a  # keep same for bootstrapping — or sample randomly
                    q_target = r + gamma * (1 - d) * target_critic(s_next, a_next)

                # compute current Q-values for critic loss
                q_current = critic(s, a)
                critic_loss = nn.MSELoss()(q_current, q_target)

                # update critic network
                optimizer_critic.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=5.0)
                optimizer_critic.step()

                # soft update conserving 0.995 of target network parameters
                for target_param, param in zip(target_critic.parameters(), critic.parameters()):
                    target_param.data.copy_(0.995 * target_param.data + 0.005 * param.data)

            # decrement movel left to progress through episode
            moves_left -= 1
            if moves_left <= 0 or done:
                break


def train(game_path, 
          max_ep_len=50, 
          _lambda=0.1, 
          lra=1e-4, lrc=1e-4, 
          batch_size=32, 
          episodes=1000, 
          gamma=0.9, 
          action_dim=3072, state_dim=3072, 
          learn_reward_shaping=False, 
          eval_interval=100, 
          train_only=False, 
          curriculum_enabled=False, 
          pretrain_critic_enabled=False, 
          random_reset=True, 
          wandb_proj=None, wandb_entity=None):
    """
    Main online training script for SAC actor model.
    See main for more param details.
    """
    # configure wandb to track training
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

    # initalize environment, LLM embeddor, actor, critic, reward shaper, target critic
    env = TextAdventureEnv(game_path)
    llama = LLaMAWrapper()
    actor = Actor(state_dim=state_dim, action_dim=action_dim)
    critic = Critic(state_dim=state_dim, action_dim=action_dim)
    reward_shaper = RewardShaper(state_dim)
    target_critic = deepcopy(critic)

    # initialize replay buffer, optimizers
    replay_buffer = ReplayBuffer()
    optimizer_actor = torch.optim.Adam(actor.parameters(), lr=lra)
    optimizer_critic = torch.optim.Adam(critic.parameters(), lr=lrc)
    optimizer_shaper = torch.optim.Adam(reward_shaper.parameters(), lr=1e-4)

    # Initialize embbeding_to_action dictionary
    embedding_to_action = {}

    # pretrain critic on random actions to boost quality of critic at start of training
    if pretrain_critic_enabled:
        pretrain_critic(env, critic, target_critic, optimizer_critic, llama, replay_buffer)

    # curriculum learning to sample from later states in game
    if curriculum_enabled:
        curriculum_max_idx = len(env.game_states) - 1
        curriculum_min_idx = 0
        curriculum_step_interval = 100

    # training loop
    print("Begin training... 🎢 ")
    for episode in tqdm(range(episodes), unit="episode"):
        # initialize episode metrics
        episode_loss_actor = 0
        episode_loss_critic = 0
        episode_reward = 0
        episode_raw_reward = 0

        # reset environment to a random point in the walkthrough
        # curriculum learning starts from later, reward rich states then steps backwards
        if curriculum_enabled:
            # NOTE: slight hack, should push to guaranteed later state
            start_idx = random.randint(curriculum_min_idx, curriculum_max_idx)
            state_text = env.reset_to_state(start_idx)
        else:
            state_text = env.reset(random_reset=random_reset)
        done = False
        moves_remaining = max_ep_len

        # progress through episode by taking actions in environment
        while not done:
            # encode current state
            prev_state_embedding = llama.encode_text(state_text).squeeze(0)

            # get action embedding from actor
            action_embedding = nn.functional.normalize(actor(prev_state_embedding), p=2, dim=-1)

            # get valid actions from environment and build embedding_to_action dict
            valid_actions = env.get_valid_actions()
            embedding_to_action = generate_embedding_action_dict(embedding_to_action, valid_actions, llama)

            # compute action text
            action_text = decode_action(action_embedding, embedding_to_action)

            # step through environment
            next_state_text, reward, done, _ = env.step(action_text)
            next_state_embedding = llama.encode_text(next_state_text).squeeze(0)

            # for learned reward shaping, use net to compute potential difference and shape reward
            if learn_reward_shaping:
                potential_diff = reward_shaper(prev_state_embedding, next_state_embedding).squeeze().clamp(min=-0.5, max=0.5)
                shaped_reward = reward + _lambda * potential_diff.detach()

            # otherwise, directly use critic to compute state value funtions
            else:
                v_current = critic(prev_state_embedding, action_embedding).mean()
                a_next = actor(next_state_embedding)
                v_next = critic(next_state_embedding, a_next).mean()
                
                # shape the reward using the critic as a potential function
                shaped_reward = reward + _lambda * (gamma * v_next.detach() - v_current.detach())

            # add the shaped reward to the replay buffer
            replay_buffer.add((prev_state_embedding.detach(), action_embedding.detach(), shaped_reward.detach(), next_state_embedding.detach(), done))

            episode_reward += shaped_reward.item()
            episode_raw_reward += reward

            # if we have enough data, sample a batch and train the critic and actor
            if len(replay_buffer.buffer) > batch_size:
                # sample a batch
                batch = replay_buffer.sample(batch_size)
                s, a, r, s_next, d = zip(*batch)
                s = torch.stack(s)
                a = torch.stack(a)
                r = torch.tensor(r).unsqueeze(1)
                s_next = torch.stack(s_next)
                d = torch.tensor(d).unsqueeze(1).float()

                # either use learned reward shaping phi OR estimate with V
                if learn_reward_shaping:
                    reward_delta = reward_shaper(s, s_next).clamp(min=-0.5, max=0.5)
                    shaped_r = r + _lambda * reward_delta.detach()
                else:
                    with torch.no_grad():
                        v_current_batch = critic(s, a).mean(dim=1, keepdim=True)
                        a_next_batch = actor(s_next)
                        v_next_batch = critic(s_next, a_next_batch).mean(dim=1, keepdim=True)
                        shaped_r = r + _lambda * (gamma * v_next_batch - v_current_batch)

                # compute target Q-values
                with torch.no_grad():
                    a_next = actor(s_next)
                    q_target = shaped_r + gamma * (1 - d) * target_critic(s_next, a_next)

                # compute current Q-values for critic loss and update critic network
                q_current = critic(s, a)
                critic_loss = nn.MSELoss()(q_current, q_target)
                optimizer_critic.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=5.0)
                optimizer_critic.step()

                # if we are using learned reward shaping, compute the shaper loss and update the shaper network (TD error)
                if learn_reward_shaping:
                    with torch.no_grad():
                        td_error = (r + gamma * (1 - d) * critic(s_next, actor(s_next)).detach() - critic(s, a).detach()).squeeze()
                    shaper_loss = nn.MSELoss()(reward_delta.squeeze(), td_error)
                    optimizer_shaper.zero_grad()
                    shaper_loss.backward()
                    torch.nn.utils.clip_grad_norm_(reward_shaper.parameters(), max_norm=5.0)
                    optimizer_shaper.step()

                # compute actor loss and update actor network
                a_pred = nn.functional.normalize(actor(s), p=2, dim=-1)
                actor_loss = -critic(s, a_pred).mean()
                optimizer_actor.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=5.0)
                optimizer_actor.step()

                # update metrics 
                episode_loss_actor += actor_loss.item()
                episode_loss_critic += critic_loss.item()

                # soft update conserving 0.995 of target critic network parameters
                for target_param, param in zip(target_critic.parameters(), critic.parameters()):
                    target_param.data.copy_(0.995 * target_param.data + 0.005 * param.data)

            # progress through episode
            state_text = next_state_text
            moves_remaining -= 1
            if moves_remaining < 0:
                done = True

        # log metrics
        wandb.log({
            "episode": episode,
            "loss_actor": episode_loss_actor,
            "loss_critic": episode_loss_critic,
            "reward": episode_reward,
            "raw_reward": episode_raw_reward,
            "q_current": q_current.mean().item(),
            "q_target": q_target.mean().item(),
            "potential_diff": reward_delta.mean().item() if learn_reward_shaping else 0.0
        })

        # when using curriculum learning, update the max start index every so often
        if curriculum_enabled and episode % curriculum_step_interval == 0:
            curriculum_max_idx = max(curriculum_max_idx - 1, curriculum_min_idx)
            print(f"Curriculum updated: max start index is now {curriculum_max_idx}")

        # save the model after training every 50 episodes
        if episode % 50 == 0:
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
    parser.add_argument('--curriculum_enabled', type=bool, default=False, help='Use curriculum learning to slowly step back game start index')
    parser.add_argument('--pretrain_critic_enabled', type=bool, default=False, help='Run initial loop to gain quality expreience for critic')
    parser.add_argument('--random_reset', type=bool, default=True, help='When resetting state, randomly choose from available states. If false, resets from beginning.')
    parser.add_argument('--max_ep_len', type=int, default=50, help='Max turns agent can take each episode')

    args = parser.parse_args()
    train(
        args.game_path,
        max_ep_len=args.max_ep_len,
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
        curriculum_enabled=args.curriculum_enabled,
        pretrain_critic_enabled=args.pretrain_critic_enabled,
        random_reset=args.random_reset,
        wandb_proj=args.wandb_proj,
        wandb_entity=args.wandb_entity
    )
