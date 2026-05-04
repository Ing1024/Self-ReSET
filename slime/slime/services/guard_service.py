"""
Guard Model HTTP Service

Standalone FastAPI server that wraps ContentEvaluator and GenLabelEvaluator,
allowing the RolloutManager to call guard models via HTTP instead of loading
them locally. This frees up GPU resources for training and rollout inference.

Usage:
    CUDA_VISIBLE_DEVICES=7 python -m slime.services.guard_service \
        --safety-stream-model-path /path/to/Qwen3Guard-Stream \
        --safety-guard-model-path /path/to/Qwen3Guard-Gen \
        --port 8100
"""

import argparse
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Guard Model Service")

# Global evaluators (initialized on startup)
_content_evaluator = None
_gen_label_evaluator = None
_executor = ThreadPoolExecutor(max_workers=2)


# ============================================================================
# Request / Response schemas
# ============================================================================

class ContentEvalRequest(BaseModel):
    messages: List[Dict[str, str]]
    thinking: bool = False


class ContentEvalResponse(BaseModel):
    pred_risk_levels: List[int]
    pred_categories: List[str]
    pred_risk_prob: List[float]
    input_ids: List[int]
    split_idx_eval: Optional[int] = None


class GenLabelRequest(BaseModel):
    prompt: str
    response: str
    label: Optional[str] = None
    thinking: bool = False


class GenLabelResponse(BaseModel):
    safety_label: Optional[str]
    refusal_label: Optional[str]


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "content_evaluator": _content_evaluator is not None,
        "gen_label_evaluator": _gen_label_evaluator is not None,
    }


@app.post("/content_evaluate_response", response_model=ContentEvalResponse)
async def content_evaluate_response(req: ContentEvalRequest):
    if _content_evaluator is None:
        raise HTTPException(status_code=503, detail="ContentEvaluator not loaded")
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        _executor,
        lambda: _content_evaluator.process_conversation_response(req.messages, thinking=req.thinking),
    )
    return ContentEvalResponse(**result)


@app.post("/content_evaluate_prompt")
async def content_evaluate_prompt(req: ContentEvalRequest):
    if _content_evaluator is None:
        raise HTTPException(status_code=503, detail="ContentEvaluator not loaded")
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        _executor,
        lambda: _content_evaluator.process_conversation_prompt(req.messages),
    )
    return result


@app.post("/gen_label_single", response_model=GenLabelResponse)
async def gen_label_single(req: GenLabelRequest):
    if _gen_label_evaluator is None:
        raise HTTPException(status_code=503, detail="GenLabelEvaluator not loaded")

    from slime.utils.types import Sample
    sample = Sample(prompt=req.prompt, response=req.response, label=req.label)

    loop = asyncio.get_event_loop()
    safety_label, refusal_label = await loop.run_in_executor(
        _executor,
        lambda: _gen_label_evaluator.gen_label_single(sample, thinking=req.thinking),
    )
    return GenLabelResponse(safety_label=safety_label, refusal_label=refusal_label)


@app.post("/gen_label_single_eval", response_model=GenLabelResponse)
async def gen_label_single_eval(req: GenLabelRequest):
    if _gen_label_evaluator is None:
        raise HTTPException(status_code=503, detail="GenLabelEvaluator not loaded")

    from slime.utils.types import Sample
    sample = Sample(prompt=req.prompt, response=req.response, label=req.label)

    loop = asyncio.get_event_loop()
    safety_label, refusal_label = await loop.run_in_executor(
        _executor,
        lambda: _gen_label_evaluator.gen_label_single_eval(sample, thinking=req.thinking),
    )
    return GenLabelResponse(safety_label=safety_label, refusal_label=refusal_label)


@app.post("/gen_label_single_eval_wildguard", response_model=GenLabelResponse)
async def gen_label_single_eval_wildguard(req: GenLabelRequest):
    if _gen_label_evaluator is None:
        raise HTTPException(status_code=503, detail="GenLabelEvaluator not loaded")

    from slime.utils.types import Sample
    sample = Sample(prompt=req.prompt, response=req.response, label=req.label)

    loop = asyncio.get_event_loop()
    safety_label, refusal_label = await loop.run_in_executor(
        _executor,
        lambda: _gen_label_evaluator.gen_label_single_eval_wildguard(sample, thinking=req.thinking),
    )
    return GenLabelResponse(safety_label=safety_label, refusal_label=refusal_label)


# ============================================================================
# Startup
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Guard Model HTTP Service")
    parser.add_argument("--safety-stream-model-path", type=str, default=None,
                        help="Path to the ContentEvaluator (stream safety) model")
    parser.add_argument("--safety-guard-model-path", type=str, default=None,
                        help="Path to the GenLabelEvaluator (guard gen) model")
    parser.add_argument("--eval-model-path", type=str, default=None,
                        help="Path to the eval model (WildGuard). Defaults to safety-guard-model-path.")
    parser.add_argument("--llama-guard-model-path", type=str, default=None,
                        help="Path to Llama Guard model (optional, for evaluation mode)")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8100)
    return parser.parse_args()


def main():
    global _content_evaluator, _gen_label_evaluator

    args = parse_args()

    # Load ContentEvaluator
    if args.safety_stream_model_path:
        logger.info(f"Loading ContentEvaluator from {args.safety_stream_model_path} ...")
        from slime.rollout.Guardforward import ContentEvaluator
        _content_evaluator = ContentEvaluator(args.safety_stream_model_path)
        logger.info("ContentEvaluator loaded.")

    # Load GenLabelEvaluator
    if args.safety_guard_model_path:
        eval_model_path = args.eval_model_path or args.safety_guard_model_path
        logger.info(f"Loading GenLabelEvaluator from {args.safety_guard_model_path} ...")
        from slime.rollout.Guardforward import GenLabelEvaluator
        _gen_label_evaluator = GenLabelEvaluator(
            args.safety_guard_model_path,
            eval_model_path,
            args.llama_guard_model_path,
        )
        logger.info("GenLabelEvaluator loaded.")

    if _content_evaluator is None and _gen_label_evaluator is None:
        logger.warning("No guard models loaded! Provide at least one model path.")

    logger.info(f"Starting guard service on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
