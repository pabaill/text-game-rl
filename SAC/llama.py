from transformers import AutoModel, AutoTokenizer
import torch

class LLaMAWrapper:
    def __init__(self, model_name="meta-llama/Llama-3.2-3B-Instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def encode_text(self, text: str):
        tokens = self.tokenizer(text, return_tensors="pt")
        outputs = self.model(**tokens)
        return outputs.last_hidden_state.mean(dim=1)  # [batch, hidden_dim]

    def decode_action(self, action_embed, action_bank_embeddings, action_bank_texts):
        # Nearest neighbor search for mapping embedding to action
        similarities = torch.cosine_similarity(action_embed, action_bank_embeddings)
        top_idx = similarities.argmax().item()
        return action_bank_texts[top_idx]