#!/bin/bash
# ============================================================================
# Guard Model HTTP Service Launcher
#
# Starts the guard model service on a specified GPU before training begins.
# The service shares the GPU with an SGLang rollout engine (they never run
# simultaneously, so memory contention is minimal).
#
# Usage:
#   bash scripts/start_guard_service.sh [GPU_ID] [PORT]
#
# Example:
#   bash scripts/start_guard_service.sh 7 8100
#
# Then in your training script, add:
#   --guard-service-url http://localhost:8100
#   --actor-num-gpus-per-node 8   (use all 8 GPUs for training+rollout)
# ============================================================================

set -e

GPU_ID=${1:-7}
PORT=${2:-8100}

# Model paths - adjust these to your setup
STREAM_MODEL="Qwen3Guard-Stream-8B"
GUARD_MODEL="Qwen3Guard-Gen-8B"
EVAL_MODEL="wildguard"
LLAMA_GUARD_MODEL="Llama-Guard-3-8B"

echo "Starting guard model service on GPU ${GPU_ID}, port ${PORT}..."

CUDA_VISIBLE_DEVICES=${GPU_ID} python -m slime.services.guard_service \
    --safety-stream-model-path "${STREAM_MODEL}" \
    --safety-guard-model-path "${GUARD_MODEL}" \
    --eval-model-path "${EVAL_MODEL}" \
    --llama-guard-model-path "${LLAMA_GUARD_MODEL}" \
    --port ${PORT} &

GUARD_PID=$!
echo "Guard service PID: ${GUARD_PID}"

# Wait for the service to be healthy
echo "Waiting for guard service to be ready..."
for i in $(seq 1 120); do
    if curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        echo "Guard service is ready on http://localhost:${PORT}"
        exit 0
    fi
    sleep 2
done

echo "ERROR: Guard service failed to start within 240 seconds"
kill ${GUARD_PID} 2>/dev/null
exit 1
