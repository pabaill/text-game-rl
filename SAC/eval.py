import pandas as pd
import torch
from torch.nn.functional import cosine_similarity
from llama import LLaMAWrapper
from ac import Actor
import argparse

def evaluate_model(actor, llama, test_data, batch_size=32, output_file="evaluation_results.csv"):
    """
    Evaluate the model by comparing predicted actions to ground truth actions using cosine similarity.
    """
    actor.eval()
    results = []
    batch_data = test_data.sample(batch_size)

    # Create action bank
    action_texts = test_data['action'].unique()
    action_embeddings = torch.stack([llama.encode_text(a).squeeze(0) for a in action_texts])

    for _, row in batch_data.iterrows():
        prev_state_text = row['state']
        expected_action_text = row['action']

        # Convert texts to embeddings
        prev_state_embedding = llama.encode_text(prev_state_text).squeeze(0)
        expected_action_embedding = llama.encode_text(expected_action_text).squeeze(0)

        # Get predicted action embedding from the actor
        predicted_action_embedding = actor(prev_state_embedding)

        # Compare to all action candidates
        similarities = cosine_similarity(predicted_action_embedding.unsqueeze(0), action_embeddings)
        closest_idx = similarities.argmax()
        predicted_action_text = action_texts[closest_idx]
        similarity = cosine_similarity(
            predicted_action_embedding.unsqueeze(0),
            expected_action_embedding.unsqueeze(0),
            dim=1
        ).item()

        # Append evaluation results
        results.append({
            "state": prev_state_text,
            "expected_action": expected_action_text,
            "predicted_action": predicted_action_text,
            "cosine_similarity": similarity
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    avg_similarity = results_df['cosine_similarity'].mean()
    print(f"Average cosine similarity to ground truth: {avg_similarity:.4f}")
    return avg_similarity

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
