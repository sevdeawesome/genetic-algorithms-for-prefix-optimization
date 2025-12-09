"""
greedy search for prefix optimization
"""
from tqdm import tqdm
from src.models import load_model, get_target_prob


def greedy_search(
    model, tokenizer, device,
    prompt: str,
    target_token: str,
    prefix_length: int = 5,
    vocab_subset: list[int] = None,
) -> tuple[list[int], float]:
    """
    greedily find prefix tokens
    """
    if vocab_subset is None:
        vocab_subset = list(range(tokenizer.vocab_size))

    prefix_ids = []

    for pos in range(prefix_length):
        best_token = None
        best_prob = -1

        for token_id in tqdm(vocab_subset, desc=f"Position {pos+1}/{prefix_length}"):
            candidate = prefix_ids + [token_id]
            prefix_text = tokenizer.decode(candidate)
            full_text = prefix_text + prompt

            prob = get_target_prob(model, tokenizer, device, full_text, target_token)

            if prob > best_prob:
                best_prob = prob
                best_token = token_id

        prefix_ids.append(best_token)
        print(f"  Best so far: {repr(tokenizer.decode(prefix_ids))} -> P={best_prob:.6f}")

    return prefix_ids, best_prob
