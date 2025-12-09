"""
Simple model wrapper to quickly get next token probability
"""
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model(model_name: str = "gpt2"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer, device


def get_next_token_probs(model, tokenizer, device, text: str) -> torch.Tensor:
    """Get the probabilty distribution over the next token"""
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]  # [vocab_size]
    return F.softmax(logits, dim=-1)


def get_target_prob(model, tokenizer, device, text: str, target_token: str) -> float:
    """Get probability of a specific token being next."""
    probs = get_next_token_probs(model, tokenizer, device, text)
    target_id = tokenizer.encode(target_token, add_special_tokens=False)[0]
    return probs[target_id].item()


if __name__ == "__main__":
    #testing it
    model, tokenizer, device = load_model("gpt2")

    prompt = "How do I make a cake?"
    print(f"Prompt: {prompt}")
    print(f"P(' Sure') = {get_target_prob(model, tokenizer, device, prompt, ' Sure'):.6f}")
    print(f"P(' No') = {get_target_prob(model, tokenizer, device, prompt, ' No'):.6f}")
