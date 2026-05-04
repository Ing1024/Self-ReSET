import asyncio
import base64
import copy
import io
from argparse import Namespace
from collections import defaultdict
from typing import Any, Callable, Optional, Union
from copy import deepcopy
import time
from typing import List

import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer
import wandb
import re
from typing import Dict, List, Tuple
from collections import Counter


from slime.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from slime.rollout.filter_hub.base_types import DynamicFilterOutput
from slime.utils.async_utils import run
from slime.utils.data import Dataset
from slime.utils.eval_config import EvalDatasetConfig
from slime.utils.http_utils import get, post
from slime.utils.mask_utils import get_response_lengths
from slime.utils.misc import SingletonMeta, load_function
from slime.utils.types import Sample

from .rm_hub import async_rm, batched_async_rm

from slime.rollout.Guardforward import ContentEvaluator, GenLabelEvaluator 

__all__ = ["generate_rollout"]

def _load_and_encode_image(path: str) -> str:
    """Load an image from path, ensure RGB, encode as JPEG base64 string."""
    with Image.open(path) as image:
        buffer = io.BytesIO()
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


class GenerateState(metaclass=SingletonMeta):
    """
    The global state for the generation process.
    """

    def __init__(self, args: Namespace) -> None:
        # persistant state for the generation process
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
        self.semaphore = asyncio.Semaphore(
            args.sglang_server_concurrency * args.rollout_num_gpus // args.rollout_num_gpus_per_engine
        )
        self.sampling_params: dict[str, Any] = dict(
            temperature=args.rollout_temperature,
            top_p=args.rollout_top_p,
            top_k=args.rollout_top_k,
            max_new_tokens=args.rollout_max_response_len,
            stop=args.rollout_stop,
            stop_token_ids=args.rollout_stop_token_ids,
            skip_special_tokens=args.rollout_skip_special_tokens,
            no_stop_trim=True,
            spaces_between_special_tokens=False,
            repetition_penalty=getattr(args, "rollout_repetition_penalty", 1.0),
        )

        print(f"[GenerateState] Using sampling_params: {self.sampling_params}", flush=True)

        if getattr(args, "sglang_enable_deterministic_inference", False):
            sampling_seed_base = args.rollout_seed
            self.group_sampling_seeds = [sampling_seed_base + i for i in range(args.n_samples_per_prompt)]

        self.reset()

    def reset(self) -> None:
        self.remaining_batch_size = 0
        self.pendings = set()
        self.aborted = False

    def submit_generate_tasks(self, samples: list[list[Sample]], content_evaluator: ContentEvaluator = None, genlabel_evaluator: GenLabelEvaluator = None) -> None:
        for group in samples:
            self.pendings.add(
                asyncio.create_task(
                    # submit a group of samples as a single task.
                    generate_and_rm_group(
                        self.args,
                        group,
                        sampling_params=self.sampling_params.copy(),
                        evaluation=False,
                        content_evaluator=content_evaluator,
                        genlabel_evaluator=genlabel_evaluator,
                    )
                )
            )
        self.remaining_batch_size += len(samples)


async def generate(args: Namespace, sample: Sample, sampling_params: dict[str, Any]) -> Sample: # generate step 6: innermost single-prompt sglang generation logic, no need to change
    """Generate using traditional SGLang router with token-based workflow"""
    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    assert (
        sample.status == Sample.Status.PENDING or sample.status == Sample.Status.ABORTED
    ), f"Sample status is {sample.status}"

    # Process prompt to create text and image payload
    image_data = []
    if isinstance(sample.prompt, str):
        text_prompt = sample.prompt
    else:  # Multimodal prompt (list of dicts)
        text_prompt = ""
        # sglang uses a placeholder to insert image features
        image_token = state.tokenizer.special_tokens_map.get("image_token", "<image>")
        for part in sample.prompt:
            if part["type"] == "text":
                text_prompt += part["text"]
            elif part["type"] == "image":
                text_prompt += image_token
                try:
                    img_b64 = await asyncio.to_thread(_load_and_encode_image, part["path"])
                    image_data.append(img_b64)
                except Exception as e:
                    print(f"Error processing image {part['path']}: {e}")
                    sample.status = Sample.Status.ABORTED
                    return sample

    # Handle prefill case: if response exists but tokens are empty, tokenize prompt + response
    if len(sample.response) > 0 and len(sample.tokens) == 0:
        # This is a prefill case: tokenize prompt + prefill response together
        # Tokenize them together to ensure correct tokenization at the boundary
        full_text = text_prompt + sample.response
        full_token_ids = state.tokenizer(full_text, add_special_tokens=False)["input_ids"]
        sample.tokens = full_token_ids
        # Adjust max_new_tokens: we've already generated the prefill, so we need to continue from there
        prompt_token_ids = state.tokenizer(text_prompt, add_special_tokens=False)["input_ids"]
        prefill_length = len(full_token_ids) - len(prompt_token_ids)
        # Initialize response_length with prefill token length
        sample.response_length = prefill_length
        
        #print(f"prefill_length: {prefill_length}", flush=True)
        
        # Update pre_rep_length to token length if it exists (convert from char length to token length)
        if hasattr(sample, 'pre_rep_length') and sample.pre_rep_length is not None:
            # pre_rep_length was stored as char length in data.py, update it to token length
            sample.pre_rep_length = prefill_length
        elif sample.metadata and 'pre_rep_length' in sample.metadata:
            # Update in metadata as well
            sample.metadata['pre_rep_length'] = prefill_length
        #sampling_params["max_new_tokens"] -= prefill_length
        #print(f"max_new_tokens: {sampling_params['max_new_tokens']}", flush=True)
    elif len(sample.response) > 0:
        # Adjust max_new_tokens for subsequent generation turns
        prompt_len = len(state.tokenizer(text_prompt, add_special_tokens=False)["input_ids"])
        #sampling_params["max_new_tokens"] -= len(sample.tokens) - prompt_len

    assert (
        sampling_params["max_new_tokens"] >= 0
    ), f"max_new_tokens: {sampling_params['max_new_tokens']} should not be less than 0"
    if sampling_params["max_new_tokens"] == 0:
        sample.status = Sample.Status.TRUNCATED
        return sample

    # Prepare payload for sglang server
    payload = {
        "sampling_params": sampling_params,
        "return_logprob": True,
    }

    if args.use_rollout_routing_replay:
        payload["return_routed_experts"] = True

    if image_data:
        payload["image_data"] = image_data

    # Use existing tokens for multi-turn or tokenize the new prompt
    if len(sample.response) > 0:
        payload["input_ids"] = sample.tokens
    else:
        prompt_token_ids = state.tokenizer(text_prompt, add_special_tokens=False)["input_ids"]
        payload["input_ids"] = prompt_token_ids
        if not sample.tokens:  # Initialize sample.tokens for the first turn
            sample.tokens = prompt_token_ids

    output = await post(url, payload)

    # Extract new response tokens

    if args.use_slime_router and "RadixTreeMiddleware" in args.slime_router_middleware_paths:
        assert not args.partial_rollout, "Currently parital rollout is not suppurted when using slime router"
        retrieve_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/retrieve_from_text"
        retrieve_payload = {"text": sample.prompt + output["text"], "return_logp": True}
        retrieve_output = await post(retrieve_url, retrieve_payload)
        sample.tokens = retrieve_output["tokens"]
        sample.response += output["text"]
        sample.loss_mask = retrieve_output["loss_mask"]
        sample.response_length = get_response_lengths([sample.loss_mask])[0]
        sample.loss_mask = sample.loss_mask[-sample.response_length :]
        sample.rollout_log_probs = retrieve_output["rollout_logp"][-sample.response_length :]
        # Notice: currently cannot get the spec info from radix router output.
    else:
        if "output_token_logprobs" in output["meta_info"]:
            new_response_tokens = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
            new_response_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
        else:
            new_response_tokens, new_response_log_probs = [], []
            
        # For prefill case: if response_length was initialized with prefill length,
        # we need to add placeholder log_probs for prefill tokens before adding new ones
        if sample.rollout_log_probs is None:
            sample.rollout_log_probs = []
        
        # Check if we have prefill tokens that need placeholder log_probs
        # This happens when response_length was set to prefill_length but rollout_log_probs is empty
        if len(sample.rollout_log_probs) == 0 and sample.response_length > 0:
            # We have prefill tokens, add placeholder log_probs for them
            # Use 0.0 as placeholder (these won't be used in training due to loss_mask)
            #print("Hello, this is to test whether we entered the flaw-thinking branch", flush=True)
            sample.rollout_log_probs = [0.0] * sample.response_length
        
        # Update sample with tokens directly - avoiding re-tokenization
        sample.tokens = sample.tokens + new_response_tokens
        sample.response_length += len(new_response_tokens)
        sample.response += output["text"]
            
        sample.rollout_log_probs += new_response_log_probs

        if args.sglang_speculative_algorithm:
            # cannot directly use spec info from sglang because of partial rollout.
            sample.spec_info.add(
                meta_info=output["meta_info"],
                response_length=sample.response_length,
            )

    if "weight_version" in output["meta_info"]:
        sample.weight_versions.append(output["meta_info"]["weight_version"])

    if "routed_experts" in output["meta_info"]:
        assert len(output["meta_info"]["routed_experts"]) == len(sample.tokens)
        sample.rollout_routed_experts = np.array(output["meta_info"]["routed_experts"])

    match output["meta_info"]["finish_reason"]["type"]:
        case "length":
            sample.status = Sample.Status.TRUNCATED
        case "abort":
            sample.status = Sample.Status.ABORTED
        case "stop":
            sample.status = Sample.Status.COMPLETED

    return sample


async def generate_and_rm( 
    args: Namespace,
    sample: Union[Sample, list[Sample]],
    sampling_params: dict[str, Any],
    evaluation: bool = False,
    content_evaluator: ContentEvaluator=None,
    genlabel_evaluator: GenLabelEvaluator=None,
) -> Union[Sample, list[Sample]]: # generate step 5: per-token assignments, enters the actual generate data-generation logic
    # For samples with existing response, check if they're complete
    if sample.status == Sample.Status.COMPLETED or sample.status == Sample.Status.TRUNCATED:
        assert sample.response is not None
        if not args.group_rm:
            assert sample.reward is not None
        return sample

    state = GenerateState(args)

    # generate
    async with state.semaphore:
        if state.aborted:
            sample.status = Sample.Status.ABORTED
            return sample

        # rollout each prompt individually
        if args.custom_generate_function_path is not None:
            custom_generate_func = load_function(args.custom_generate_function_path)
            sample = await custom_generate_func(args, sample, sampling_params)
        else:
            sample = await generate(args, sample, sampling_params)

    # for the rm that need the whole group, we will not do the rm here
    if args.group_rm:
        return sample

    # If this is a single-prompt reward, continue; this is the default state for GRPO-like experiments

    # multi samples
    if isinstance(sample, list):
        samples = sample
        if any([sample.status == Sample.Status.ABORTED for sample in samples]):
            return samples

        # for multi agent system, the reward of some sample is calculated during generation.
        samples_need_reward = [sample for sample in samples if sample.reward is None]
        rewards = await batched_async_rm(args, samples_need_reward)
        for sample, reward in zip(samples_need_reward, rewards):
            sample.reward = reward
        return samples
    else:
        if sample.status == Sample.Status.ABORTED:
            return sample
        #print("Yes , I am here", flush=True)
        sample.reward = await async_rm(args, sample, content_evaluator=content_evaluator, genlabel_evaluator=genlabel_evaluator, evaluation=evaluation)
        # print(f"reward: {sample.reward}", flush=True)
        # print(f"sample.label: {sample.label}, sample.safety_label: {sample.safety_label}, sample.refusal_label: {sample.refusal_label}", flush=True)
    return sample


async def generate_and_rm_group(
    args: Namespace, group: list[Sample], sampling_params: dict[str, Any], evaluation: bool = False, content_evaluator: ContentEvaluator=None, genlabel_evaluator: GenLabelEvaluator=None
) -> list[Sample]: # generate step 4: group-level reward computation (if needed), and per-prompt rollout task submission
    state = GenerateState(args)

    if state.aborted:
        return group

    tasks = []
    for idx, sample in enumerate(group):
        current_sampling_params = sampling_params.copy()
        if getattr(args, "sglang_enable_deterministic_inference", False):
            seed = state.group_sampling_seeds[idx]
            current_sampling_params["sampling_seed"] = seed
        tasks.append(generate_and_rm(args, sample, current_sampling_params, evaluation=evaluation, content_evaluator=content_evaluator, genlabel_evaluator=genlabel_evaluator)) # rollout each prompt in the group separately

    group = await asyncio.gather(*tasks)

    # for the rm that need the whole group, we will not do the rm here
    if not state.aborted and args.group_rm:
        rewards = await batched_async_rm(args, group) # compute reward at the group level
        for sample, reward in zip(group, rewards):
            sample.reward = reward

    return group


async def abort(args: Namespace, rollout_id: int) -> list[list[Sample]]:
    aborted_samples = []

    state = GenerateState(args)
    assert not state.aborted
    state.aborted = True
    response = await get(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/list_workers")

    # abort all the requests
    for url in response["urls"]:
        print(f"Abort request for {url}", flush=True)
        await post(f"{url}/abort_request", {"abort_all": True})

    # make sure all the pending tasks are finished
    count = 0
    while state.pendings:
        done, state.pendings = await asyncio.wait(state.pendings, return_when=asyncio.FIRST_COMPLETED)

        if not args.partial_rollout:
            continue

        # for partial rollout, collect the partial samples into the data buffer
        for task in done:
            group = task.result()
            for sample in group:
                if sample.response and "start_rollout_id" not in sample.metadata:
                    sample.metadata["start_rollout_id"] = rollout_id
            aborted_samples.append(group)
            count += len(group)

    if args.partial_rollout:
        print(f"Collected {count} partial samples into the data buffer", flush=True)

    return aborted_samples


def visualize_token_labels(eval_pred):
    """Visualize each token and its streamlabel"""
    tokens = eval_pred.get("response_tokens", [])
    risk_levels = eval_pred.get("pred_risk_levels", [])
    categories = eval_pred.get("pred_categories", [])
    risk_probs = eval_pred.get("pred_risk_prob", [])

    # risk-level mapping
    risk_level_map = {0: "safe", 1: "controversial", 2: "unsafe"}

    print("\n" + "="*100)
    print("Token-Label visualization")
    print("="*100)
    print(f"{'Idx':<6} {'Token':<30} {'RiskLevel':<12} {'RiskProb':<12} {'Category':<40}")
    print("-"*100)

    for i, (token, risk_level, category, risk_prob) in enumerate(zip(tokens, risk_levels, categories, risk_probs)):
        # handle special characters for terminal readability
        token_display = repr(token) if token in ['\n', '\t', ' '] or len(token.strip()) == 0 else token
        token_display = token_display[:28]  # limit length
        risk_level_name = risk_level_map.get(risk_level, f"Unknown({risk_level})")
        print(f"{i:<6} {token_display:<30} {risk_level_name:<12} {risk_prob:.4f}      {category:<40}")

    print("="*100)
    print(f"\nTotal: {len(tokens)} tokens")
    print("\nRisk-level counts:")
    risk_counter = Counter(risk_levels)
    for level, count in sorted(risk_counter.items()):
        print(f"  {risk_level_map.get(level, f'Unknown({level})')}: {count} tokens")

    print("\nCategory counts:")
    category_counter = Counter(categories)
    for category, count in sorted(category_counter.items(), key=lambda x: -x[1]):
        print(f"  {category}: {count} tokens")

async def safety_check_label_single(
    args: Namespace,
    sample: Sample,
    content_evaluator: ContentEvaluator,
    thinking: bool = False
) -> tuple[Sample, str]:
    """
    Run safety detection on a single sample. If two consecutive unsafe or
    controversial tokens are detected, truncate the response at the first
    unsafe/controversial position.

    Args:
        sample: Sample object to check; contains prompt and response.
        content_evaluator: ContentEvaluator instance used for safety evaluation.
        thinking: whether thinking tags are included, default False.

    Returns:
        Sample: the truncated sample or None. If no unsafe content is detected,
        returns None; if unsafe content is detected, returns the truncated
        sample (status set to TRUNCATED).
    """
    # convert prompt into messages format
    if isinstance(sample.prompt, str):
        # if prompt is a string, convert to message format
        # assume prompt is user input and response is assistant reply
        prompt=sample.prompt
        response=sample.response
        if(prompt.startswith("<｜begin▁of▁sentence｜><｜User｜>")):
            prompt = prompt[len("<｜begin▁of▁sentence｜><｜User｜>"):]
        if(prompt.endswith("<｜Assistant｜><think>\n")):
            prompt = prompt[:-len("<｜Assistant｜><think>\n")]
        if(prompt.endswith("<｜Assistant｜>")):
            prompt = prompt[:-len("<｜Assistant｜>")]
        if(response.endswith("<｜end▁of▁sentence｜>")):
            response = response[:-len("<｜end▁of▁sentence｜>")]

        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]
    # elif isinstance(sample.prompt, list):
    #     # if prompt is already a message list, use it directly
    #     messages = deepcopy(sample.prompt)
    #     # ensure the last message is the assistant response
    #     if messages and messages[-1].get("role") != "assistant":
    #         messages.append({"role": "assistant", "content": sample.response})
    #     else:
    #         # if the last one is already assistant, update its content
    #         messages[-1]["content"] = sample.response
    else:
        raise ValueError(f"Unsupported prompt type: {type(sample.prompt)}")

    # print(f"Safety check single: Here is messages {messages}", flush=True)
    # run safety evaluation with ContentEvaluator
    eval_result = content_evaluator.process_conversation_response(messages, thinking=thinking)

    # if(sample.label == "unsafe"and sample.reward == 0.0):
    #     print(
    #         f"rollout sample example case after check: {[str(sample.prompt) + sample.response]}",
    #         flush=True
    #     )
    #     visualize_token_labels(eval_result)
    # print(f"Safety check single: Here is eval_result {eval_result}", flush=True)

    pred_risk_levels = eval_result["pred_risk_levels"]
    split_idx_eval = eval_result["split_idx_eval"]
    input_ids = eval_result["input_ids"]

    # with probability args.label_flip, flip the label
    for i in range(len(pred_risk_levels)):
        if getattr(args, 'label_flip', 0.0) > 0.0:
            import random
            if random.random() < args.label_flip:
                pred_risk_levels[i] = 0 if pred_risk_levels[i] != 0 else 1

    # check for two consecutive unsafe (1) or controversial (2) tokens
    has_consecutive_unsafe = False
    first_two_unsafe_idx = None
    for i in range(len(pred_risk_levels) - 1):
        current_level = pred_risk_levels[i]
        next_level = pred_risk_levels[i + 1]

        # check for two consecutive unsafe or controversial tokens
        if (current_level != 0 and next_level != 0):
            has_consecutive_unsafe = True
            first_two_unsafe_idx = i
            break

    if has_consecutive_unsafe:
        label = "unsafe"
    else:
        label = "safe"

    # # with probability args.label_flip, flip the label
    # if getattr(args, 'label_flip', 0.0) > 0.0:
    #     import random
    #     if random.random() < args.label_flip:
    #         label = "safe" if label == "unsafe" else "unsafe"
    #         print(f"Label flipped to: {label}", flush=True)

    # if consecutive unsafe content is detected, find the first unsafe/controversial position
    first_unsafe_idx = None
    if has_consecutive_unsafe:
        for j in range(len(pred_risk_levels)):
            if pred_risk_levels[j] == 1 or pred_risk_levels[j] == 2:
                first_unsafe_idx = j
                break

        if args.enable_unsafe2_truncate:
            first_unsafe_idx = first_two_unsafe_idx

    # when enable_noresp is set, check whether a </think> tag appears before first_unsafe_idx
    # if so, the unsafe content is in the resp part and should not be truncated
    if args.enable_noresp and first_unsafe_idx is not None:
        tokenizer = content_evaluator.tokenizer
        # decode the tokens between split_idx_eval and first_unsafe_idx
        tokens_before_unsafe = input_ids[split_idx_eval:split_idx_eval + first_unsafe_idx]
        text_before_unsafe = tokenizer.decode(tokens_before_unsafe, skip_special_tokens=False)
        if "</think>" in text_before_unsafe:
            first_unsafe_idx = None
            print(f"resp unsafe case, dropped", flush=True)

    # if no unsafe content was detected, return None
    if first_unsafe_idx is None:
        return None, label

    if label == "safe":
        return None, label

    # if this prompt was already rolled, skip truncation
    if sample.partialrollout:
        return None, label

    # compute the truncation index within the full input_ids
    # pred_risk_levels starts from split_idx_eval, so add this offset
    # truncate before the first unsafe/controversial token (inclusive)
    truncate_token_idx = split_idx_eval + first_unsafe_idx

    # print(f"label: {label}", flush=True)
    # print(f"first_unsafe_idx: {first_unsafe_idx}", flush=True)
    # print(f"truncate_token_idx: {truncate_token_idx}", flush=True)
    # print(f"partialrollout: {sample.partialrollout}", flush=True)

    # find a full sentence: the first punctuation after the first unsafe/controversial token (whole sentences are more coherent)
        # define CJK and ASCII punctuation
    import string
    punctuation_chars = set(string.punctuation) | {'。', '！', '？', '，', '；', '：', '、', '…', '—', '「', '」', '『', '』', '（', '）', '【', '】', '《', '》'}

    tokenizer = content_evaluator.tokenizer
    # search forward from first_unsafe_idx for the first punctuation
    punctuation_token_idx = None
    max_search_length = min(len(pred_risk_levels) - first_unsafe_idx, 100)  # look ahead at most 100 tokens to avoid infinite loops

    # decode up to the first unsafe token as the baseline
    base_tokens = input_ids[split_idx_eval:split_idx_eval + first_unsafe_idx + 1]
    base_text = tokenizer.decode(base_tokens, skip_special_tokens=False)

    for offset in range(1, max_search_length):  # start searching after the first unsafe token
        current_token_idx = split_idx_eval + first_unsafe_idx + offset
        if current_token_idx >= len(input_ids):
            break

        # decode tokens from split_idx_eval up to current_token_idx
        tokens_to_decode = input_ids[split_idx_eval:current_token_idx + 1]
        decoded_text = tokenizer.decode(tokens_to_decode, skip_special_tokens=False)

        # check whether the newly added text contains punctuation
        # compare current decoded text to base text to find the newly added slice
        if len(decoded_text) > len(base_text):
            new_text = decoded_text[len(base_text):]
            # check whether the newly added text contains punctuation
            for char in new_text:
                if char in punctuation_chars:
                    # found punctuation; use the current token position
                    punctuation_token_idx = current_token_idx
                    # print(f"decoded_text: {decoded_text}", flush=True)
                    # print(f"new_text: {new_text}", flush=True)
                    # print(f"char: {char}", flush=True)
                    break

        if punctuation_token_idx is not None:
            break

    # if a punctuation was found, use its position; otherwise use the first unsafe token
    if(args.enable_punctuation_after):
        if punctuation_token_idx is not None :
            #print("Found the punctuation token; using the punctuation token", flush=True)
            truncate_token_idx = punctuation_token_idx
        else:
            truncate_token_idx = split_idx_eval + first_unsafe_idx
    else:
        truncate_token_idx = split_idx_eval + first_unsafe_idx
    #print(f"truncate_token_idx: {truncate_token_idx}", flush=True)
    # truncate the response at the token position
    # decode tokens from split_idx_eval up to truncate_token_idx (inclusive)
    tokenizer = content_evaluator.tokenizer
    response_token_ids = input_ids[split_idx_eval:truncate_token_idx + 1] # +1 includes the token
    truncated_response = tokenizer.decode(response_token_ids, skip_special_tokens=False)

    # build the truncated sample
    truncated_sample = deepcopy(sample)
    truncated_sample.response = truncated_response
    truncated_sample.response_length = len(response_token_ids)
    truncated_sample.status = Sample.Status.ABORTED # for compatibility, set directly to ABORTED
    truncated_sample.partialrollout=True
    truncated_sample.in_buffer=True
    truncated_sample.partial_response = truncated_response # record the previous truncation for case inspection
    if(args.enable_partial_mask):
        #print("partial but masked", flush=True)
        truncated_sample.pre_rep_length = len(response_token_ids) # reuse recap logic to mask it

    prompt_token_count = len(truncated_sample.tokens) - sample.response_length
    # update tokens (if present)
    if truncated_sample.tokens:
        # compute how many tokens to keep
        # tokens should contain prompt tokens + response tokens
        # assume the head of tokens is prompt and the tail is response
        truncated_sample.tokens = (
            truncated_sample.tokens[:prompt_token_count] + response_token_ids
        )

    # update loss_mask (if present)
    if truncated_sample.loss_mask:
        truncated_sample.loss_mask = truncated_sample.loss_mask[:truncated_sample.response_length]
    
    if truncated_sample.rollout_log_probs is not None:
        truncated_sample.rollout_log_probs = truncated_sample.rollout_log_probs[:truncated_sample.response_length]

    if truncated_sample.rollout_routed_experts is not None:
        truncated_sample.rollout_routed_experts = truncated_sample.rollout_routed_experts[:prompt_token_count + truncated_sample.response_length]
    
    if not args.enable_add_after:
        return truncated_sample, label
    
    # append a "wait" token after the punctuation break to trigger model reflection
    # compatible with SFT-style log prob and mask handling
    wait_token_text = "wait"
    wait_token_ids = tokenizer.encode(wait_token_text, add_special_tokens=False)
    wait_token_count = len(wait_token_ids)

    # save the current response_length (before appending the wait token)
    current_response_length = truncated_sample.response_length

    # append the wait token to tokens
    truncated_sample.tokens = truncated_sample.tokens + wait_token_ids

    # update response_length (add the wait token count)
    truncated_sample.response_length += wait_token_count

    # update response text (append the wait token)
    truncated_sample.response = truncated_response + wait_token_text

    # update loss_mask: add mask entries for the wait token (1 = compute loss)
    if truncated_sample.loss_mask is None:
        truncated_sample.loss_mask = [1] * current_response_length
    truncated_sample.loss_mask = truncated_sample.loss_mask + [1] * wait_token_count # 0 = skip, 1 = compute

    # update rollout_log_probs: add log probs for the wait token (SFT-style: 0.0)
    if truncated_sample.rollout_log_probs is None:
        truncated_sample.rollout_log_probs = [0.0] * current_response_length
    truncated_sample.rollout_log_probs = truncated_sample.rollout_log_probs + [0.0] * wait_token_count

    # update rollout_routed_experts: add routed-expert info for the wait token (empty list = no routed expert)
    if truncated_sample.rollout_routed_experts is not None:
        import numpy as np
        # add routed-expert info for the wait token using empty lists (matches list[list[int]] type)
        # if rollout_routed_experts is a numpy array, extend it
        if isinstance(truncated_sample.rollout_routed_experts, np.ndarray):
            # create an array with the same length as wait_token_count, filled with empty lists
            wait_experts = np.array([[]] * wait_token_count, dtype=object)
            truncated_sample.rollout_routed_experts = np.concatenate([truncated_sample.rollout_routed_experts, wait_experts])
        else:
            # if it's a list, append empty lists directly
            truncated_sample.rollout_routed_experts = truncated_sample.rollout_routed_experts + [[]] * wait_token_count

    # NOTE: for partialrollout samples, subsequent partial_rollout will grow
    # response_length but will not update loss_mask automatically (the generate
    # function in partial_rollout.py does not touch loss_mask). Setting loss_mask
    # to None here lets _convert_samples_to_train_data recreate an all-ones mask
    # based on the final response_length, avoiding length mismatches.
    truncated_sample.loss_mask = None
    
    # record truncation info in metadata
    # truncated_sample.metadata["safety_truncated"] = True
    # truncated_sample.metadata["truncate_token_idx"] = truncate_token_idx
    # truncated_sample.metadata["first_unsafe_token_idx"] = first_unsafe_idx
    # truncated_sample.metadata["pred_risk_level"] = pred_risk_levels[first_unsafe_idx]
    
    return truncated_sample, label


async def safety_check_label(
    args: Namespace,
    data: list[list[Sample]],
    rollout_id: int,
    content_evaluator: ContentEvaluator,
    genlabel_evaluator: GenLabelEvaluator,
) -> list[Sample]:
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
            If safety_guard_model_path is missing or None, return None.

    Returns:
        list[Sample] or None: processed data in the same format as the input.
        If unsafe content is detected, the corresponding sample is truncated
        and its status is set to TRUNCATED; otherwise return None.
    """
    # check whether a guard model is configured
    guard_model_path = getattr(args, "safety_stream_model_path", None)
    if guard_model_path is None:
        # no guard model configured, return original data
        return None
    
    filtered_samples = []
    need_in_buffer = 0
    timer = time.time()
    
    def _is_zero_std(samples: List[Sample]):
        rewards = [sample.get_reward_value(args) for sample in samples]
        return len(rewards) == 0 or all(rewards[0] == r for r in rewards)
    
    print(f"Here is len of data: {len(data)}", flush=True)
    print(f"Here is len of group: {len(data[0])}", flush=True)
    count_unsafe = 0
    count_refusal = 0
    for index, group in enumerate(data):
        #print(f"Here is group {index} of {len(data)}", flush=True)
        if _is_zero_std(group):
            continue
        prompt_sample= Sample(prompt=group[0].prompt, label=group[0].label, partialrollout=True) 
        #prompt_sample= Sample(prompt=group[0].prompt, label=group[0].label, partialrollout=False, in_buffer=True) 
        for index, sample in enumerate(group): 
            #print(f"Here is sample {index} of {len(group)}", flush=True)      
            # skip detection if response is empty
            if not sample.response:
                continue
            # run safety detection on the sample
            try:
                safety_label, refusal_label = genlabel_evaluator.gen_label_single(sample, thinking=args.enable_safety_label_thinking) # generate the overall label for the single prompt

                filtered_sample, _ = await safety_check_label_single(
                    args=args,
                    sample=sample,
                    content_evaluator=content_evaluator,
                    thinking=args.enable_safety_thinking,
                )# process unsafe samples filter

                # filtering logic
                if (args.refusal_in_buffer) and (refusal_label == "Yes") and (sample.label == "safe"):
                    need_in_buffer += 1 # count refusal rollouts for sample filtering

                if filtered_sample is not None :
                    #print(f"filtered_sample: {filtered_sample}", flush=True)
                    if args.stream_unsafe_final_safe and safety_label != "Safe": # str(safety_label).lower() != "safe"
                        continue
                    if filtered_sample.response and "start_rollout_id" not in filtered_sample.metadata:
                        filtered_sample.metadata["start_rollout_id"] = rollout_id
                    filtered_samples.append(filtered_sample)
                    count_unsafe += 1
            except Exception as e:
                # on error during detection, drop the sample instead of keeping the original
                print(f"Warning: Safety filter failed for sample {sample.index}: {e}", flush=True)
            if(len(sample.rollout_log_probs) != sample.response_length):
                response_length = len(sample.rollout_log_probs)
                sample.response_length = response_length
                sample.response = sample.response[:response_length]
                sample.tokens = sample.tokens[:response_length]
                if sample.loss_mask is not None:
                    sample.loss_mask = sample.loss_mask[:response_length]
                if sample.rollout_log_probs is not None:
                    sample.rollout_log_probs = sample.rollout_log_probs[:response_length]
                if sample.rollout_routed_experts is not None:
                    sample.rollout_routed_experts = sample.rollout_routed_experts[:response_length]
        if need_in_buffer > args.refusal_in_buffer_threshold: # threshold for refusal; default is 0
            filtered_samples.append(prompt_sample)
            count_refusal += 1
        need_in_buffer = 0

    print(f"Partial Count of unsafe: {count_unsafe},Partial Count of refusal: {count_refusal}", flush=True)
    #print(f"Safety check time: {time.time() - timer}", flush=True)
    return filtered_samples


async def generate_rollout_async( 
    args: Namespace, rollout_id: int, data_source: Callable[[int], list[list[Sample]]], content_evaluator: ContentEvaluator, genlabel_evaluator: GenLabelEvaluator  
) -> tuple[RolloutFnTrainOutput, list[Sample]]: # generate step 3: async task dispatch; group-level rollout entry
    """An example to implement the generate_rollout function for an rule based rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        data_source: the data source to fetch

    Returns:
        tuple[RolloutFnTrainOutput, list[Sample]]:
            - data: a list of groups of samples generated by the rollout, length equals `rollout_batch_size`
            - process_unsafe_samples: any unsafe samples collected during safety check
    """
    assert args.rollout_global_dataset

    state = GenerateState(args)

    # instantiate data filters
    dynamic_filter = (
        load_function(args.dynamic_sampling_filter_path) if args.dynamic_sampling_filter_path is not None else None
    )

    metric_gatherer = _MetricGatherer()

    # target_data_size is the total number of valid samples to get
    target_data_size = args.rollout_batch_size

    data = []
    do_print = True
    do_print_partialrollout = True
    pbar = tqdm(total=target_data_size * args.n_samples_per_prompt, desc="Rollout generation")
    epoch_count = 0
    data_count = 0
    while len(data) < target_data_size:
        while state.remaining_batch_size < target_data_size:
            # get samples from the buffer and submit the generation requests.
            samples = data_source(args.over_sampling_batch_size)
            state.submit_generate_tasks(samples, content_evaluator=content_evaluator, genlabel_evaluator=genlabel_evaluator) # submit rollout tasks: for each group in samples, submit generate_and_rm_group separately

        # wait for the generation to finish
        done, state.pendings = await asyncio.wait(state.pendings, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            group: list[Sample] = task.result()

            if do_print:
                sample = group[0][0] if isinstance(group[0], list) else group[0]
                print(
                    f"First rollout sample: {[str(sample.prompt)+sample.response]}, label: {sample.label}, reward: {sample.reward}",
                    flush=True,
                )
                do_print = False
            
            # partialrollout case output
            if do_print_partialrollout:
                sample = group[0][0] if isinstance(group[0], list) else group[0]
                if sample.partialrollout:
                    # print(f"Partial rollout sample.prompt: {sample.prompt}", flush=True)
                    # print(f"Partial rollout sample.response: {sample.response}", flush=True)
                    print(
                        f"Partial rollout sample example case before: {[str(sample.prompt) + sample.partial_response]}",
                        flush=True
                    )
                    print(
                        f"Partial rollout sample example case after: {[str(sample.prompt) + sample.response]}, label: {sample.label}, reward: {sample.reward}",
                        flush=True
                    )
                    do_print_partialrollout = False
            
            assert len(group) == args.n_samples_per_prompt
            dynamic_filter_output = _call_dynamic_filter(dynamic_filter, args, group)
            if not dynamic_filter_output.keep:
                metric_gatherer.on_dynamic_filter_drop(reason=dynamic_filter_output.reason)
                state.remaining_batch_size -= 1
                continue    
            
            # def _is_zero_std(samples: List[Sample]):
            #     rewards = [sample.get_reward_value(args) for sample in samples]
            #     return len(rewards) == 0 or all(rewards[0] == r for r in rewards)
            
            # if args.enable_dynamic_filter and _is_zero_std(group): # dapo dynamic sampling when std=0
            #     metric_gatherer.on_dynamic_filter_drop(reason="zero_std")
            #     state.remaining_batch_size -= 1
            #     continue

            # add the samples to the data
            # NOTE: here we have not stored all the unused samples back to the data buffer.
            if len(data) < target_data_size:
                data.append(group)
                pbar.update(args.n_samples_per_prompt)
                data_count += 1
                sample = group[0][0] if isinstance(group[0], list) else group[0]
                if (not sample.partialrollout):
                    epoch_count += 1

    pbar.close()
    sample = data[-1][0][0] if isinstance(data[-1][0], list) else data[-1][0] # multiturn logic branch
    print(
        f"Finish rollout: {[str(sample.prompt) + sample.response]}, label: {sample.label}, reward: {sample.reward}",
        flush=True,
    )
    print(f"data_count: {data_count}, epoch_count: {epoch_count}", flush=True)
    # there are still some unfinished requests, abort them
    aborted_samples = await abort(args, rollout_id) # get aborted samples; no train-data conversion needed

    assert len(data) == args.rollout_batch_size, f"Got {len(data)} samples, expected {args.rollout_batch_size}"
    data = sorted(data, key=lambda group: group[0][0].index if isinstance(group[0], list) else group[0].index)

    # add a safety-check step here
    if args.enable_partialrollout:
        process_unsafe_samples =await safety_check_label(args, data, rollout_id, content_evaluator, genlabel_evaluator)  # custom: pass the content evaluator
    else:
        process_unsafe_samples = []

    # reset the global state to prevent effects on the next rollout or eval.
    state.reset()
    return RolloutFnTrainOutput(samples=data, metrics=metric_gatherer.collect()), process_unsafe_samples


def _call_dynamic_filter(fn, *args, **kwargs):
    if fn is None:
        return DynamicFilterOutput(keep=True)

    output = fn(*args, **kwargs)

    # compatibility for legacy version
    if not isinstance(output, DynamicFilterOutput):
        output = DynamicFilterOutput(keep=output)

    return output


class _MetricGatherer:
    def __init__(self):
        self._dynamic_filter_drop_reason_count = defaultdict(lambda: 0)

    def on_dynamic_filter_drop(self, reason: Optional[str]):
        if not reason:
            return
        self._dynamic_filter_drop_reason_count[reason] += 1

    def collect(self):
        return {
            f"rollout/dynamic_filter/drop_{reason}": count
            for reason, count in self._dynamic_filter_drop_reason_count.items()
        }


EVAL_PROMPT_DATASET = {}


async def eval_rollout(args: Namespace, rollout_id: int, content_evaluator: ContentEvaluator, genlabel_evaluator: GenLabelEvaluator) -> tuple[dict[str, dict[str, list[Any]]], list[list[Sample]]]:
    assert not args.group_rm, "Group RM is not supported for eval rollout"
    results = {}
    for dataset_cfg in getattr(args, "eval_datasets", []) or []:
        results.update(await eval_rollout_single_dataset(args, rollout_id, dataset_cfg, content_evaluator, genlabel_evaluator))
    return RolloutFnEvalOutput(data=results), []


async def eval_rollout_single_dataset(
    args: Namespace, rollout_id: int, dataset_cfg: EvalDatasetConfig, content_evaluator: ContentEvaluator, genlabel_evaluator: GenLabelEvaluator
) -> dict[str, dict[str, list[Any]]]:
    """An example to implement the eval_rollout function for an rule based rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        dataset_cfg: configuration of the dataset
    """
    assert not args.group_rm, "Group RM is not supported for eval rollout"

    global EVAL_PROMPT_DATASET

    name = dataset_cfg.name
    path = dataset_cfg.path

    def _resolve_dataset_setting(dataset_value, eval_value, rollout_value):
        if dataset_value is not None:
            return dataset_value
        if eval_value is not None:
            return eval_value
        return rollout_value

    prompt_key = _resolve_dataset_setting(
        dataset_cfg.prompt_key,
        args.eval_input_key,
        args.input_key,
    )
    label_key = _resolve_dataset_setting(
        dataset_cfg.label_key,
        args.eval_label_key,
        args.label_key,
    )
    tool_key = _resolve_dataset_setting(
        dataset_cfg.tool_key,
        args.eval_tool_key,
        args.tool_key,
    )
    metadata_key = dataset_cfg.metadata_key or getattr(args, "metadata_key", "metadata")

    cache_key = dataset_cfg.cache_key + (args.hf_checkpoint, args.apply_chat_template)
    if cache_key not in EVAL_PROMPT_DATASET:
        tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
        EVAL_PROMPT_DATASET[cache_key] = Dataset(
            path,
            tokenizer=tokenizer,
            max_length=args.rollout_max_prompt_len,
            prompt_key=prompt_key,
            label_key=label_key,
            multimodal_keys=args.multimodal_keys,
            metadata_key=metadata_key,
            tool_key=tool_key,
            apply_chat_template=args.apply_chat_template,
            apply_chat_template_kwargs=args.apply_chat_template_kwargs,
        )
    dataset = EVAL_PROMPT_DATASET[cache_key]

    base_sampling_params = dict(
        temperature=_resolve_dataset_setting(dataset_cfg.temperature, args.eval_temperature, args.rollout_temperature),
        top_p=_resolve_dataset_setting(
            dataset_cfg.top_p,
            args.eval_top_p,
            args.rollout_top_p,
        ),
        top_k=_resolve_dataset_setting(
            dataset_cfg.top_k,
            args.eval_top_k,
            args.rollout_top_k,
        ),
        max_new_tokens=_resolve_dataset_setting(
            dataset_cfg.max_response_len,
            args.eval_max_response_len,
            args.rollout_max_response_len,
        ),
        stop=dataset_cfg.stop if dataset_cfg.stop is not None else args.rollout_stop,
        stop_token_ids=(
            dataset_cfg.stop_token_ids if dataset_cfg.stop_token_ids is not None else args.rollout_stop_token_ids
        ),
        skip_special_tokens=args.rollout_skip_special_tokens,
        no_stop_trim=True,
        spaces_between_special_tokens=False,
        repetition_penalty=_resolve_dataset_setting(
            getattr(dataset_cfg, "repetition_penalty", None),
            getattr(args, "eval_repetition_penalty", None),
            getattr(args, "rollout_repetition_penalty", 1.2),
        ),
    )

    print(f"[eval_rollout] Using sampling_params for dataset '{name}': {base_sampling_params}", flush=True)

    min_new_tokens = dataset_cfg.min_new_tokens
    if min_new_tokens is None:
        min_new_tokens = getattr(args, "eval_min_new_tokens", None)
    if min_new_tokens is not None:
        base_sampling_params["min_new_tokens"] = min_new_tokens

    n_samples_per_prompt = (
        dataset_cfg.n_samples_per_eval_prompt
        if dataset_cfg.n_samples_per_eval_prompt is not None
        else args.n_samples_per_eval_prompt
    )

    tasks = []
    # do multiple samples for eval prompts
    sample_index = 0
    for i, prompt_sample in enumerate(dataset.samples):
        for j in range(n_samples_per_prompt):
            # use the same prompt for multiple samples
            sample = copy.deepcopy(prompt_sample)
            sample.index = sample_index
            sample_index += 1
            sample.metadata = dataset_cfg.inject_metadata(getattr(sample, "metadata", None))
            sampling_params = base_sampling_params
            if getattr(args, "sglang_enable_deterministic_inference", False):
                sampling_params = base_sampling_params.copy()
                sampling_params["sampling_seed"] = args.rollout_seed + j
            #print(f"max_new_tokens: {sampling_params['max_new_tokens']}", flush=True)
            tasks.append(
                generate_and_rm(
                    args,
                    sample,
                    sampling_params=sampling_params,
                    content_evaluator=content_evaluator,
                    genlabel_evaluator=genlabel_evaluator,
                    evaluation=True,
                )
            )

    data = []
    do_print = 0
    pbar = tqdm(total=len(tasks), desc="Rollout generation", disable=not do_print)
    for coro in asyncio.as_completed(tasks):
        sample = await coro
        if do_print<5:
        #if do_print<10000:
            safety_label, refusal_label = genlabel_evaluator.gen_label_single(sample, thinking=args.enable_safety_label_thinking) # generate overall label for the single prompt

            filtered_sample, process_safety_label = await safety_check_label_single(
                args=args,
                sample=sample,
                content_evaluator=content_evaluator,
                thinking=args.enable_safety_thinking,
            )# process unsafe samples filter
            print(
                "eval_rollout_single_dataset example data prompt:",
                [str(sample.prompt)],
                "eval_rollout_single_dataset example data response:",
                [str(sample.response)],
                f"reward={sample.reward}",
                f"label={sample.label}",
                f"safety_label={safety_label}",
                f"process_safety_label={process_safety_label}",
                f"refusal_label={refusal_label}",
                f"do_print={do_print}",
                flush=True,
            )
            do_print += 1
        if isinstance(sample, list):
            data.extend(sample)
        else:
            data.append(sample)
        pbar.update(1)
    pbar.close()

    data.sort(key=lambda sample: sample.index)

    reward_key = args.eval_reward_key or args.reward_key
    return {
        name: {
            "rewards": [sample.reward if not reward_key else sample.reward[reward_key] for sample in data],
            "truncated": [sample.status == Sample.Status.TRUNCATED for sample in data],
            "samples": data,
        }
    }


# TODO remove this temp function
def generate_rollout( 
    args: Namespace, rollout_id: int, data_buffer: Any, content_evaluator: ContentEvaluator, genlabel_evaluator: GenLabelEvaluator, evaluation: bool = False
) -> Union[RolloutFnTrainOutput, RolloutFnEvalOutput]: # generate step 1: public entry point; returns the final output and (aborted) samples to put back in the buffer
    """An example to implement the generate_rollout function for an rule based rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        data_buffer: the data buffer to store the generated samples
        evaluation: bool, whether the rollout is for evaluation or not

    Returns:
        list[list[Sample]]: a list of list of samples generated by the rollout
    """
    output, process_unsafe_samples = generate_abortable_samples(
        args, rollout_id, data_buffer.get_samples, content_evaluator, genlabel_evaluator, evaluation=evaluation # data_buffer.get_samples: prefer the buffer, fall back to the source dataset if the buffer is short
    )
    print(f"Here is num of process_unsafe_samples {len(process_unsafe_samples)}", flush=True)
    if args.use_wandb:
        wandb.log({
            "rollout_id": rollout_id,
            "rollout/num_process_unsafe_samples": len(process_unsafe_samples),
        })
    
    data_buffer.custom_add_samples_to_buffer(process_unsafe_samples)
    return output


def generate_abortable_samples( # generate step 2: second-level dispatch entry; decides whether to run rollout or eval
    args: Namespace,
    rollout_id: int,
    data_source: Callable[[int], list[list[Sample]]],
    content_evaluator: ContentEvaluator,
    genlabel_evaluator: GenLabelEvaluator,
    evaluation: bool = False,
) -> tuple[Any, list[Sample]]:
    assert args.rollout_global_dataset
    if evaluation:
        return run(eval_rollout(args, rollout_id, content_evaluator, genlabel_evaluator))
    return run(generate_rollout_async(args, rollout_id, data_source, content_evaluator, genlabel_evaluator))
    # training_data, data = run(generate_rollout_async(args, rollout_id, data_source, content_evaluator))
    # process_unsafe_samples = safety_check(args, data, rollout_id, content_evaluator)
    # return training_data, process_unsafe_samples