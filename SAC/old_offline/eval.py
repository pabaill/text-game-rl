"""
Evaluation script for OFFLINE trained actor model.
Note: not used in final implementation, which uses online evaluation.
"""

import pandas as pd
import torch
from torch import nn
from torch.nn.functional import cosine_similarity
from llama import LLaMAWrapper
from ac import Actor
import argparse

BATCH_SIZE = 100

def evaluate_model(actor, llama, test_data, batch_size=BATCH_SIZE, output_file="evaluation_results.csv"):
    """
    Evaluate the model by comparing predicted actions to ground truth actions using cosine similarity.
    Logs embedding information for debugging purposes.
    """
    actor.eval()
    results = []
    batch_data = test_data.sample(batch_size, replace=True)

    # Create action bank
    action_texts = test_data['action'].unique()
    action_embeddings = torch.stack([nn.functional.normalize(llama.encode_text(a).squeeze(0), p=2, dim=-1) for a in action_texts])

    for idx, row in batch_data.iterrows():
        prev_state_text = row['state']
        expected_action_text = row['action']

        # Convert texts to embeddings
        prev_state_embedding = llama.encode_text(prev_state_text).squeeze(0)
        expected_action_embedding = llama.encode_text(expected_action_text).squeeze(0)

        # Get actor output before and after normalization
        raw_output = actor(prev_state_embedding)

        # NOTE: if we want to normalize the output, uncomment below.
        # predicted_action_embedding = nn.functional.normalize(raw_output, p=2, dim=-1)
        predicted_action_embedding = raw_output

        # Logging for debugging
        print(f"[Sample {idx}] Prev state embedding (first 5 dims): {prev_state_embedding[:5]}")
        print(f"[Sample {idx}] Raw actor output (first 5 dims): {raw_output[:5]}")
        print(f"[Sample {idx}] Normalized actor output (first 5 dims): {predicted_action_embedding[:5]}")
        print(f"[Sample {idx}] Expected action embedding (first 5 dims): {expected_action_embedding[:5]}")
        print(f"[Sample {idx}] Vector norms - Predicted: {predicted_action_embedding.norm():.4f}, Expected: {expected_action_embedding.norm():.4f}")

        
        # NOTE: if we want to compare via cosine similarity or euclidian distance, uncomment below.
        # Compare to all action candidates
        # similarities = cosine_similarity(predicted_action_embedding.unsqueeze(0), action_embeddings)
        # closest_idx = similarities.argmax()
        # predicted_action_text = action_texts[closest_idx]
        # similarity = cosine_similarity(
        #     predicted_action_embedding.unsqueeze(0),
        #     expected_action_embedding.unsqueeze(0),
        #     dim=1
        # ).item()

        # euclidian_dist = torch.linalg.norm(action_embeddings - predicted_action_embedding.unsqueeze(0), dim=1)
        # closest_idx = euclidian_dist.argmin()
        # predicted_action_text = action_texts[closest_idx]
        # similarity = euclidian_dist[closest_idx].item()

        # For now, use dot product to get similarity
        similarity = torch.matmul(action_embeddings, predicted_action_embedding)
        predicted_action_text = action_texts[similarity.argmax()]

        # Append evaluation results
        results.append({
            "state": prev_state_text,
            "expected_action": expected_action_text,
            "predicted_action": predicted_action_text,
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    return 

def run_evaluation(csv_path, actor_ckpt_path, state_dim, action_dim, output_file="evaluation_results.csv"):
    data = pd.read_csv(csv_path)
    llama = LLaMAWrapper()
    actor = Actor(state_dim=state_dim, action_dim=action_dim)
    actor.load_state_dict(torch.load(actor_ckpt_path, map_location=torch.device('cpu')))
    evaluate_model(actor, llama, data, output_file=output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained actor model.")
    parser.add_argument('--input_csv', type=str, required=True, help="Path to the CSV dataset to run inference on")
    parser.add_argument('--actor_ckpt_path', type=str, required=True, help="Path to the actor model checkpoint")
    parser.add_argument('--state_dim', type=int, default=3072, help="State embedding dimension")
    parser.add_argument('--action_dim', type=int, default=3072, help="Action embedding dimension")
    parser.add_argument('--output_file', type=str, help="CSV path to save evaluation results in.")

    args = parser.parse_args()
    run_evaluation(
        csv_path=args.input_csv,
        actor_ckpt_path=args.actor_ckpt_path,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        output_file=args.output_file
    )
