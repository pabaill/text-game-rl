import pandas as pd
import torch
from torch import nn
from torch.nn.functional import cosine_similarity
from replay import ReplayBuffer
from shaper import RewardShaper
from llama import LLaMAWrapper
from ac import Actor, Critic
from copy import deepcopy
import wandb
import argparse
from tqdm import tqdm

def test_action_generation(actor, llama, test_data, batch_size=32, output_file="test_results.csv"):
    """
    Test the generation of actions during a playthrough and save the results to a CSV file.
    """

    actor.eval()
    results = []
    batch_data = test_data.sample(batch_size)

    # Create action bank
    action_texts = test_data['action'].unique()
    action_embeddings = torch.stack([llama.encode_text(a).squeeze(0) for a in action_texts])

    for index, row in batch_data.iterrows():
        prev_state_text = row['state']
        expected_action_text = row['action']  # This is the ground truth action

        # Convert state and action to embeddings
        prev_state_embedding = llama.encode_text(prev_state_text).squeeze(0)
        expected_action_embedding = llama.encode_text(expected_action_text).squeeze(0)

        # Get the action predicted by the agent
        # predicted_action_embedding = actor(prev_state_embedding)
        
        # Decode the predicted action from the embedding
        # predicted_action_text = llama.decode_text(predicted_action_embedding)

        # similarity = cosine_similarity(predicted_action_embedding.unsqueeze(0), expected_action_embedding.unsqueeze(0)).item()

        # Find closest action in action bank
        similarities = torch.nn.functional.cosine_similarity(
            predicted_action_embedding.unsqueeze(0), 
            action_embeddings
        )
        closest_idx = similarities.argmax()
        predicted_action_text = action_texts[closest_idx]
        
        similarity = similarities[closest_idx].item()

        # Append the results to the list
        results.append({
            "state": prev_state_text,
            "expected_action": expected_action_text,
            "predicted_action": predicted_action_text,
            "cosine_similarity": similarity
        })

    # Create a DataFrame from the results list
    results_df = pd.DataFrame(results)

    # Save the results to a CSV file
    results_df.to_csv(output_file, index=False)

    # Calculate and return the accuracy
    avg_similarity = results_df['cosine_similarity'].mean()
    print(f"Batch average cosine similarity: {avg_similarity:.4f}")

    return avg_similarity

def evaluate(actor, critic, llama, data, batch_size=32, gamma=0.9, _lambda=0.1):
    actor.eval()  # Set to evaluation mode
    critic.eval()
    
    total_reward = 0
    total_loss_actor = 0
    total_loss_critic = 0
    
    # Sample a batch from the validation data
    batch_data = data.sample(batch_size)
    
    with torch.no_grad():
        for index, row in batch_data.iterrows():
            prev_state_text = row['state']
            action_text = row['action']
            reward = row['reward']
            next_state_text = row['next_state']
            done = row['done']

            prev_state_embedding = llama.encode_text(prev_state_text).squeeze(0)
            next_state_embedding = llama.encode_text(next_state_text).squeeze(0)
            action_embedding = llama.encode_text(action_text).squeeze(0)

            # Compute the Q-values and actor losses
            v_current = critic(prev_state_embedding, action_embedding).mean()
            a_next = actor(next_state_embedding)
            v_next = critic(next_state_embedding, a_next).mean()

            shaped_reward = reward + _lambda * (gamma * v_next - v_current)

            # Compute losses for actor and critic
            critic_loss = nn.MSELoss()(v_current, shaped_reward)

            a_pred = actor(prev_state_embedding)
            actor_loss = -critic(prev_state_embedding, a_pred).mean()

            total_loss_actor += actor_loss.item()
            total_loss_critic += critic_loss.item()
            total_reward += shaped_reward.item()
    
    # Return the average evaluation results
    avg_reward = total_reward / batch_size
    avg_loss_actor = total_loss_actor / batch_size
    avg_loss_critic = total_loss_critic / batch_size

    actor.train()  # Set back to training mode
    critic.train()

    return avg_reward, avg_loss_actor, avg_loss_critic

def train(csv_path, _lambda=0.1, lra=1e-4, lrc=1e-4, batch_size=32, episodes=1000, gamma=0.9, action_dim=3072, state_dim=3072, learn_reward_shaping=False, eval_interval=100, train_only=False, wandb_proj=None, wandb_entity=None):
    # Initialize WandB for logging
    wandb.init(project=wandb_proj, entity=wandb_entity)  # Replace with your W&B username and project name
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

    # load dataset
    data = pd.read_csv(csv_path)

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
    # action_texts = data['chosen_action'].unique()  # Assuming 'chosen_action' column contains all possible actions
    # action_embeddings = torch.stack([llama.encode_text(a).squeeze(0) for a in action_texts])


    for episode in tqdm(range(episodes), unit="episode"):
        episode_loss_actor = 0
        episode_loss_critic = 0
        episode_reward = 0
        episode_raw_reward = 0

        # Sampling a batch of data from the CSV file for training
        batch_data = data.sample(batch_size)

        # track episodes on console
        print(f"Episode num {episode}")
        
        for index, row in batch_data.iterrows():
            prev_state_text = row['state']
            action_text = row['action']
            reward = row['reward']
            next_state_text = row['next_state']
            done = row['done']

            prev_state_embedding = llama.encode_text(prev_state_text).squeeze(0)
            next_state_embedding = llama.encode_text(next_state_text).squeeze(0)

            # NOTE: why are we re-embedding and not using the pre-encoded action embeddings?
            # could we do: action_embedding = action_embeddings[action_texts.index(action_text)]?
            action_embedding = llama.encode_text(action_text).squeeze(0)

            if learn_reward_shaping:
                # Use net to learn reward shaping method
                potential_diff = reward_shaper(prev_state_embedding, next_state_embedding).squeeze()  # Assuming shaper outputs a scalar
                shaped_reward = reward + _lambda * potential_diff
            else:
                # Compute state value functions
                v_current = critic(prev_state_embedding, action_embedding).mean()
                a_next = actor(next_state_embedding)
                v_next = critic(next_state_embedding, a_next).mean()

                # Shape the reward using the critic as a potential function with discount gamma
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
        
        if not train_only and episode % eval_interval == 0:
            eval_reward, eval_loss_actor, eval_loss_critic = evaluate(actor, critic, llama, data, batch_size=batch_size, gamma=gamma, _lambda=_lambda)
            wandb.log({
                "episode": episode,
                "eval_reward": eval_reward,
                "eval_loss_actor": eval_loss_actor,
                "eval_loss_critic": eval_loss_critic
            })
            avg_sim = test_action_generation(actor, llama, data, batch_size=batch_size, output_file=f"checkpoints/ckpt_{episode}_results.csv")
            wandb.log({
                "episode": episode,
                "average_similarity": avg_sim
            })

        wandb.log({
            "episode": episode,
            "loss_actor": episode_loss_actor,
            "loss_critic": episode_loss_critic,
            "reward": episode_reward,
            "raw_reward": episode_raw_reward
        })
        print(f"    reward = {reward}")

        if episode % 100 == 0:
            # Save the model after training
            torch.save(actor.state_dict(), f"checkpoints/actor_model_ckpt_{episode}.pth")
            torch.save(critic.state_dict(), f"checkpoints/critic_model_ckpt_{episode}.pth")
            torch.save(reward_shaper.state_dict(), f"checkpoints/reward_shaper_ckpt_{episode}.pth")
            wandb.save(f"actor_model_ckpt_{episode}.pth")
            wandb.save(f"critic_model_ckpt_{episode}.pth")
            wandb.save(f"reward_shaper_ckpt_{episode}.pth")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train RL agent from CSV dataset.")
    parser.add_argument('--csv', type=str, required=True, help='Path to the CSV file with the dataset')
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
        csv_path=args.csv,
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
