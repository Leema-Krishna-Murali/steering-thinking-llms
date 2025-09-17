import re
import torch
import utils


ESCALATION_PATTERNS = [
    r"for educational purposes",
    r"hypothetical(?:ly)?",
    r"step[- ]?by[- ]?step",
    r"won'?t break rules|ignore rules|bypass",
    r"role[- ]?play",
    r"just confirm|already know the steps",
    r"obfuscate|encode|substitut",
    r"provide more details|be specific|exactly how",
]


def score_crescendo_risk(text: str) -> float:
    """Heuristic risk score in [0,1]."""
    text_lower = text.lower()
    hits = sum(bool(re.search(p, text_lower)) for p in ESCALATION_PATTERNS)
    # Cap and normalize roughly by number of patterns
    return min(1.0, hits / max(1, len(ESCALATION_PATTERNS) // 2))


def should_refuse(text: str, threshold: float = 0.5) -> bool:
    return score_crescendo_risk(text) >= threshold


def score_crescendo_risk_history(history_messages) -> float:
    """Aggregate multi-turn risk: higher weight to recent turns and compounding for repeated patterns."""
    if not history_messages:
        return 0.0
    weights = []
    scores = []
    # Emphasize later turns
    for idx, msg in enumerate(history_messages):
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        text = msg.get("content", "")
        s = score_crescendo_risk(text)
        scores.append(s)
        # linearly increasing weight (1..N)
        weights.append(len(history_messages) - idx)
    if not weights:
        return 0.0
    base = sum(w * s for w, s in zip(weights, scores)) / sum(weights)
    # Compound if there are multiple risky hits
    repeats = sum(1 for s in scores if s >= 0.5)
    compounded = min(1.0, base + 0.1 * max(0, repeats - 1))
    return compounded


def safe_generate(model, tokenizer, feature_vectors, model_name, prompt_or_history, max_new_tokens=512, steer_coeff=1.0):
    """
    Generate with safety-aware steering: steer toward 'refusal' when high risk,
    mildly toward 'policy-check' when medium risk, else no steering.
    """
    if isinstance(prompt_or_history, list):
        # Full conversation
        risk = score_crescendo_risk_history(prompt_or_history)
        messages_for_template = prompt_or_history
    else:
        # Single message
        msg = prompt_or_history if isinstance(prompt_or_history, dict) else {"role": "user", "content": str(prompt_or_history)}
        risk = score_crescendo_risk(msg.get("content", ""))
        messages_for_template = [msg]

    input_ids = tokenizer.apply_chat_template(messages_for_template, add_generation_prompt=True, return_tensors="pt")
    device = next(model.parameters()).device if hasattr(model, 'parameters') else (input_ids.device if hasattr(input_ids, 'device') else torch.device('cpu'))
    input_ids = input_ids.to(device)

    if risk >= 0.75 and 'refusal' in feature_vectors and 'refusal' in utils.steering_config[model_name]:
        outputs = utils.custom_generate_steering(
            model, tokenizer, input_ids,
            max_new_tokens=max_new_tokens,
            label='refusal',
            feature_vectors=feature_vectors,
            steering_config=utils.steering_config[model_name],
            steer_positive=True,
        )
    elif risk >= 0.4 and 'policy-check' in feature_vectors and 'policy-check' in utils.steering_config[model_name]:
        # Nudge policy-check positively, nudge adding-knowledge negatively
        outputs = utils.custom_generate_steering(
            model, tokenizer, input_ids,
            max_new_tokens=max_new_tokens,
            label='policy-check',
            feature_vectors=feature_vectors,
            steering_config=utils.steering_config[model_name],
            steer_positive=True,
        )
    else:
        outputs = utils.custom_generate_steering(
            model, tokenizer, input_ids,
            max_new_tokens=max_new_tokens,
            label='none',
            feature_vectors=None,
            steering_config=utils.steering_config[model_name],
            steer_positive=None,
        )

    return outputs

