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


def safe_generate(model, tokenizer, feature_vectors, model_name, prompt_message, max_new_tokens=512, steer_coeff=1.0):
    """
    Generate with safety-aware steering: steer toward 'refusal' when high risk,
    mildly toward 'policy-check' when medium risk, else no steering.
    """
    history_text = prompt_message.get("content", "") if isinstance(prompt_message, dict) else str(prompt_message)
    risk = score_crescendo_risk(history_text)

    input_ids = tokenizer.apply_chat_template([prompt_message], add_generation_prompt=True, return_tensors="pt")
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

