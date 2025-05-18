from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def complete_text(model, tokenizer, prompt, max_length=50, temperature=0.7, top_k=50, top_p=0.95):
    # Encode the input prompt text into token IDs
    inputs = tokenizer(prompt, return_tensors="pt")

    # Move tensors to the device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Generate text completions (decoding the model's output tokens)
    output = model.generate(
        inputs['input_ids'], 
        max_length=max_length + len(inputs['input_ids'][0]),  # total max length
        temperature=temperature,   # Control randomness
        top_k=top_k,               # Top-k sampling
        top_p=top_p,               # Nucleus sampling (top-p)
        do_sample=True,            # Enable sampling (instead of greedy decoding)
        num_return_sequences=1,    # Generate 1 sequence
        pad_token_id=tokenizer.eos_token_id  # Handle EOS token properly
    )

    # Decode the generated tokens back into text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

def main():
    model_name = "nvidia/Llama-3_1-Nemotron-Ultra-253B-v1"
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    res = complete_text(model, tokenizer, "There is a mailbox in front of you.")
    print(res)



if __name__ == "__main__":
    main()