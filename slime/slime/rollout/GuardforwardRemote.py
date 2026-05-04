"""
Remote Guard Model Evaluator Clients

Drop-in replacements for ContentEvaluator and GenLabelEvaluator that call
an external HTTP service instead of loading models locally. This allows the
RolloutManager to run with 0 GPUs while still performing safety evaluation.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple

import requests

from slime.utils.types import Sample

logger = logging.getLogger(__name__)


class RemoteContentEvaluator:
    """HTTP client with the same interface as ContentEvaluator."""

    def __init__(self, service_url: str, timeout: float = 120, max_retries: int = 3):
        self.service_url = service_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self._session = requests.Session()
        logger.info(f"RemoteContentEvaluator initialized, service_url={self.service_url}")

    def process_conversation_response(self, messages: List[Dict], thinking: bool = False) -> Dict:
        payload = {"messages": messages, "thinking": thinking}
        return self._post("/content_evaluate_response", payload)

    def process_conversation_prompt(self, messages: List[Dict]) -> Dict:
        payload = {"messages": messages, "thinking": False}
        return self._post("/content_evaluate_prompt", payload)

    def _post(self, endpoint: str, payload: dict) -> dict:
        url = f"{self.service_url}{endpoint}"
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._session.post(url, json=payload, timeout=self.timeout)
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                logger.warning(f"[RemoteContentEvaluator] {endpoint} attempt {attempt}/{self.max_retries} failed: {e}")
                if attempt < self.max_retries:
                    time.sleep(2 * attempt)
                else:
                    raise


class RemoteGenLabelEvaluator:
    """HTTP client with the same interface as GenLabelEvaluator."""

    def __init__(self, service_url: str, timeout: float = 120, max_retries: int = 3):
        self.service_url = service_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self._session = requests.Session()
        logger.info(f"RemoteGenLabelEvaluator initialized, service_url={self.service_url}")

    def gen_label_single(self, sample: Sample, thinking: bool = False) -> Tuple[Optional[str], Optional[str]]:
        return self._call_gen_label("/gen_label_single", sample, thinking)

    def gen_label_single_eval(self, sample: Sample, thinking: bool = False) -> Tuple[Optional[str], Optional[str]]:
        return self._call_gen_label("/gen_label_single_eval", sample, thinking)

    def gen_label_single_eval_wildguard(self, sample: Sample, thinking: bool = False) -> Tuple[Optional[str], Optional[str]]:
        return self._call_gen_label("/gen_label_single_eval_wildguard", sample, thinking)

    def _call_gen_label(self, endpoint: str, sample: Sample, thinking: bool) -> Tuple[Optional[str], Optional[str]]:
        payload = {
            "prompt": sample.prompt if isinstance(sample.prompt, str) else str(sample.prompt),
            "response": sample.response if isinstance(sample.response, str) else str(sample.response),
            "label": sample.label,
            "thinking": thinking,
        }
        result = self._post(endpoint, payload)
        return result.get("safety_label"), result.get("refusal_label")

    def _post(self, endpoint: str, payload: dict) -> dict:
        url = f"{self.service_url}{endpoint}"
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._session.post(url, json=payload, timeout=self.timeout)
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                logger.warning(f"[RemoteGenLabelEvaluator] {endpoint} attempt {attempt}/{self.max_retries} failed: {e}")
                if attempt < self.max_retries:
                    time.sleep(2 * attempt)
                else:
                    raise
