"""
Safety filtering utility for detecting and truncating unsafe response content.
Based on the ContentEvaluator implementation in streameval.py.

Usage examples:
    # Option 1: single sample filtering
    from safety_filter import filter_unsafe_response
    from slime.utils.types import Sample
    from streameval import ContentEvaluator

    evaluator = ContentEvaluator(model_path="path/to/guard/model")
    sample = Sample(prompt="user question", response="model response content")
    filtered_samples = filter_unsafe_response(sample, evaluator)

    # Option 2: batch filtering (for rollout)
    from argparse import Namespace
    from safety_filter import filter_unsafe_responses

    args = Namespace(safety_guard_model_path="path/to/guard/model", safety_thinking=False)
    data = [[sample1, sample2], [sample3, sample4]]  # list[list[Sample]]
    filtered_data = filter_unsafe_responses(data, args)
"""
from copy import deepcopy
from typing import List, Optional
from argparse import Namespace

from slime.utils.types import Sample
from Guardforward import ContentEvaluator


def filter_unsafe_response(
    sample: Sample,
    content_evaluator: ContentEvaluator,
    thinking: bool = False,
) -> Sample:
    """
    Run safety detection on a single sample. If two consecutive unsafe or
    controversial tokens are detected, truncate the response at the first
    unsafe/controversial position.

    Args:
        sample: Sample object to check; contains prompt and response.
        content_evaluator: ContentEvaluator instance used for safety evaluation.
        thinking: whether thinking tags are included, default False.

    Returns:
        Sample: the truncated sample if unsafe content is detected (with
        status set to TRUNCATED); None if no unsafe content is detected.
    """
    # convert prompt into messages format
    if isinstance(sample.prompt, str):
        # if prompt is a string, convert to message format
        # assume prompt is user input and response is assistant reply
        messages = [
            {"role": "user", "content": sample.prompt},
            {"role": "assistant", "content": sample.response}
        ]
    elif isinstance(sample.prompt, list):
        # if prompt is already a message list, use directly
        messages = deepcopy(sample.prompt)
        # ensure the last message is the assistant response
        if messages and messages[-1].get("role") != "assistant":
            messages.append({"role": "assistant", "content": sample.response})
        else:
            # if the last one is already assistant, update its content
            messages[-1]["content"] = sample.response
    else:
        raise ValueError(f"Unsupported prompt type: {type(sample.prompt)}")

    # run safety evaluation with ContentEvaluator
    eval_result = content_evaluator.process_conversation_response(messages, thinking=thinking)

    pred_risk_levels = eval_result["pred_risk_levels"]
    split_idx_eval = eval_result["split_idx_eval"]
    input_ids = eval_result["input_ids"]

    # check for two consecutive unsafe (1) or controversial (2) tokens
    has_consecutive_unsafe = False
    for i in range(len(pred_risk_levels) - 1):
        current_level = pred_risk_levels[i]
        next_level = pred_risk_levels[i + 1]

        # check for two consecutive unsafe or controversial tokens
        if (current_level != 0 and next_level != 0):
            has_consecutive_unsafe = True
            break

    # if consecutive unsafe content is detected, find the first unsafe/controversial position
    first_unsafe_idx = None
    if has_consecutive_unsafe:
        for j in range(len(pred_risk_levels)):
            if pred_risk_levels[j] == 1 or pred_risk_levels[j] == 2:
                first_unsafe_idx = j
                break

    # if no unsafe content was detected, return None
    if first_unsafe_idx is None:
        return None

    # compute truncation index within the full input_ids
    # pred_risk_levels starts from split_idx_eval, so add this offset
    # truncate before the first unsafe/controversial token (exclusive)
    truncate_token_idx = split_idx_eval + first_unsafe_idx

    # truncate the response at the token position
    # decode tokens from split_idx_eval up to (but not including) truncate_token_idx
    tokenizer = content_evaluator.tokenizer
    response_token_ids = input_ids[split_idx_eval:truncate_token_idx]
    truncated_response = tokenizer.decode(response_token_ids, skip_special_tokens=False)

    # build the truncated sample
    truncated_sample = deepcopy(sample)
    truncated_sample.response = truncated_response
    truncated_sample.response_length = len(response_token_ids)
    truncated_sample.status = Sample.Status.TRUNCATED

    # update tokens (if present)
    if truncated_sample.tokens:
        # compute how many tokens to keep
        # tokens should contain prompt tokens + response tokens
        # assume the head of tokens is prompt, and the tail is response
        prompt_token_count = len(truncated_sample.tokens) - sample.response_length
        truncated_sample.tokens = (
            truncated_sample.tokens[:prompt_token_count] + response_token_ids
        )

    # update loss_mask (if present)
    if truncated_sample.loss_mask:
        prompt_mask_count = len(truncated_sample.loss_mask) - sample.response_length
        truncated_sample.loss_mask = (
            truncated_sample.loss_mask[:prompt_mask_count] + [1] * len(response_token_ids)
        )

    # record truncation info in metadata
    truncated_sample.metadata["safety_truncated"] = True
    truncated_sample.metadata["truncate_token_idx"] = truncate_token_idx
    truncated_sample.metadata["first_unsafe_token_idx"] = first_unsafe_idx
    truncated_sample.metadata["pred_risk_level"] = pred_risk_levels[first_unsafe_idx]

    return truncated_sample


def filter_unsafe_responses(
    data: List[List[Sample]],
    args: Namespace,
) -> List[Sample]:
    """
    Run batch safety detection and truncation on rollout-generated data.
    If two consecutive unsafe or controversial tokens are detected, truncate
    the response at the first unsafe/controversial position.

    Args:
        data: list[list[Sample]]. Rollout-generated data; each inner list is
            a group containing multiple Samples.
        args: Namespace with the following attributes:
            - safety_guard_model_path (str, optional): path to the guard model.
            - safety_thinking (bool, optional): whether thinking tags are
              included, default False.
            If safety_guard_model_path is missing or None, return the original
            data untouched.

    Returns:
        List[List[Sample]]: processed data in the same format as the input.
        If unsafe content is detected, the corresponding sample is truncated
        and its status is set to TRUNCATED.
    """
    # check whether a guard model is configured
    guard_model_path = getattr(args, "safety_guard_model_path", None)
    if guard_model_path is None:
        # no guard model configured, return original data
        return data

    thinking = getattr(args, "safety_thinking", False)

    # initialize ContentEvaluator lazily (only when needed)
    # use singleton pattern to avoid re-initialization
    if not hasattr(filter_unsafe_responses, "_evaluator"):
        filter_unsafe_responses._evaluator = ContentEvaluator(guard_model_path)

    content_evaluator = filter_unsafe_responses._evaluator

    # iterate over all sample groups
    filtered_data = []
    for group in data:
        for sample in group:
            # skip detection if response is empty
            if not sample.response:
                continue

            # run safety detection on the sample
            try:
                filtered_sample = filter_unsafe_response(
                    sample=sample,
                    content_evaluator=content_evaluator,
                    thinking=thinking,
                )
                # filter_unsafe_response returns Sample or None
                if filtered_sample is not None:
                    filtered_data.append(filtered_sample)
            except Exception as e:
                # on error during detection, keep the original sample
                print(f"Warning: Safety filter failed for sample {sample.index}: {e}", flush=True)
                filtered_data.append(sample)

    return filtered_data

