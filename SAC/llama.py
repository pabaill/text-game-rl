"""
Wrapper for LLaMA model to encode text and decode actions.
See https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct for more details.
"""

from transformers import AutoModel, AutoTokenizer
import torch

class LLaMAWrapper:
    def __init__(self, model_name="meta-llama/Llama-3.2-3B-Instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).eval()

    def encode_text(self, text: str):
        tokens = self.tokenizer(text, return_tensors="pt")
        outputs = self.model(**tokens)
        return outputs.last_hidden_state.mean(dim=1)  # [batch, hidden_dim=3072]

    def decode_action(self, action_embed, action_bank_embeddings, action_bank_texts):
        # Nearest neighbor search for mapping embedding to action
        similarities = torch.cosine_similarity(action_embed, action_bank_embeddings)
        top_idx = similarities.argmax().item()
        return action_bank_texts[top_idx]