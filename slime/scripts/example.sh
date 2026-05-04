pkill -9 -f "sglang" || true
sleep 3
ray stop --force || true
pkill -9 -f "ray::" || true                     
pkill -9 -f "train_epoch.py" || true
sleep 3
pkill -9 -f "ray::" || true
pkill -9 -f "train_epoch.py" || true

set -ex

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16
   
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/qwen2.5-7B.sh" # change the model script according to your needs

CKPT_ARGS=(
   --hf-checkpoint $BASE_DIR/DeepSeek-R1-Distill-Qwen-7B # change the hf-checkpoint path according to your needs
   #--hf-checkpoint /root/Qwen3-4B-FP8
   --ref-load $BASE_DIR/DS-Qwen-7B_torch_dist # change the ref-load path according to your needs
   --load $BASE_DIR/DS-Qwen-7B_Self_RESET  
   --save $BASE_DIR/DS-Qwen-7B_Self_RESET
   --save-interval 10
)

ROLLOUT_ARGS=(
   --guard-service-url http://localhost:8100 # please first run the start_guard_service.sh script to start the guard service
   --enable-noresp
   --enable-partial-mask
   --buffer-max-length 256
   --enable-partialrollout 
   --enable-safety-thinking 
   --prompt-data $BASE_DIR/data/train/vanilla_filter_balanced_1500.jsonl
   --input-key prompt
   --label-key label
   #--safety-model-num-gpus 1
   --safety-stream-model-path Qwen3Guard-Stream-8B # set real model path to train
   --safety-guard-model-path Qwen3Guard-Gen-8B
   #--eval-model-path wildguard
   --llama-guard-model-path Llama-Guard-3-8B
   --use-llama-guard

   --rollout-function-path slime.rollout.partialrollout.generate_rollout
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-epoch 4
   --rollout-batch-size 64
   --n-samples-per-prompt 16
   --rollout-max-response-len 8192
   --rollout-temperature 0.7
   --rollout-top-p 1.0
   --rollout-top-k -1
   --rollout-repetition-penalty 1.0

   # enable dynamic sampling
   --over-sampling-batch-size 64
   --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std

   --global-batch-size 1024
   --balance-data
)

CUSTOM_ARGS=(
   --custom-rm-path slime.rollout.rm_hub.safety_rm.reward_func
)

EVAL_ARGS=(
   --eval-interval 5
   --eval-prompt-data wildjailbreak $BASE_DIR/data/wildjailbreak/wildjailbreak_converted.jsonl XSTest $BASE_DIR/data/evaluation/overrefusal/XSTest.jsonl
   --eval-function-path slime.rollout.partialrollout.generate_rollout
   --n-samples-per-eval-prompt 2
   --eval-max-response-len 8000
   --eval-top-p 1.0
   --eval-temperature 0.0
   --eval-top-k -1
   --eval-repetition-penalty 1.2
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

# WANDB_ARGS=(
#    # --use-wandb
# )

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.7
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash
)

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# Build the runtime environment JSON with proper variable substitution
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 $BASE_DIR/train_epoch.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   --colocate \
   --num-workers 2 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${CUSTOM_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]}