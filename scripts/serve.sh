#!/usr/bin/env bash
# 双 RTX 3090 上的 27B 稠密模型：W4A8 + DFlash2 + 192K 上下文
#
#   MODEL=~/models/qwen3.8-27b-autoround-int4 \
#   DRAFT=~/models/dflash2-w4a16 \
#   API_KEY=sk-xxx bash serve.sh
#
# 关键环境变量见文件末尾的说明。
set -u

AI="${AI:-$HOME/ai}"
VENV="${VENV:-$AI/venv}"
MODEL="${MODEL:?需要 MODEL 指向 AutoRound INT4 权重目录}"
DRAFT="${DRAFT:-$AI/models/dflash2-w4a16}"

CTX="${CTX:-196608}"      # 192K
SEQ="${SEQ:-16}"          # max_num_seqs
MBT="${MBT:-2048}"        # max_num_batched_tokens
SN="${SN:-7}"             # 投机草稿深度，合法域 [5,7]
KVOFF="${KVOFF:-16}"      # KV offload 到主机的 GiB，硬上限 16
RET="${RET:-448}"         # = tokens_per_chunk 的整数倍
PORT="${PORT:-8000}"
SERVED="${SERVED:-q27}"
API_KEY="${API_KEY:-}"

# --- 关键环境变量（每一个都有实测依据，见 docs/DEPLOY.md）---
export VLLM_USE_FLASHINFER_SAMPLER=0     # sm86 上 FlashInfer 采样器会挂
export NCCL_CUMEM_ENABLE=0
unset PYTORCH_CUDA_ALLOC_CONF            # 与 KV offload 的 madvise 冲突
export OMP_NUM_THREADS=1
export VLLM_NO_USAGE_STATS=1
export VLLM_MARLIN_INPUT_DTYPE=int8       # <<< W4A8 的开关（INT8 激活）
export VLLM_MARLIN_USE_ATOMIC_ADD=1
export VLLM_MARLIN_INT8_EXCLUDE_RE="${EXCL:-lm_head|mtp}"   # 这两个层必须留在 BF16
[ -n "${INCL:-}" ] && export VLLM_MARLIN_INT8_INCLUDE_RE="$INCL"

export CUDA_HOME="${CUDA_HOME:-$VENV/lib/python3.12/site-packages/nvidia/cu13}"
export PATH="$VENV/bin:$CUDA_HOME/bin:$PATH"

# 若打过 fp8-KV 插件，需要它的运行时路径；没有则跳过
[ -f "$AI/tools/fa2-artifacts/fa2-runtime.env" ] && source "$AI/tools/fa2-artifacts/fa2-runtime.env"

SPECARG=()
[ "${NOSPEC:-0}" = "1" ] || SPECARG=(--speculative-config \
  "{\"method\":\"dflash\",\"model\":\"$DRAFT\",\"num_speculative_tokens\":$SN,\"draft_sample_method\":\"probabilistic\"}")

AUTHARG=()
[ -n "$API_KEY" ] && AUTHARG=(--api-key "$API_KEY")

exec "$VENV/bin/vllm" serve "$MODEL" \
  --served-model-name "$SERVED" \
  --quantization auto_round --dtype bfloat16 \
  --tensor-parallel-size 2 \
  --max-model-len "$CTX" \
  --gpu-memory-utilization "${GPU_UTIL:-0.93}" \
  --max-num-seqs "$SEQ" \
  --max-num-batched-tokens "$MBT" \
  --kv-cache-dtype "${KVD:-auto}" \
  --kv-offloading-size "$KVOFF" --kv-offloading-backend native \
  --mamba-ssm-cache-dtype bfloat16 \
  --enable-prefix-caching --enable-chunked-prefill \
  --prefix-cache-retention-interval "$RET" \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  "${AUTHARG[@]}" \
  --host 0.0.0.0 --port "$PORT" \
  "${SPECARG[@]}"

# ── 参数速查 ────────────────────────────────────────────────────────────────
# CTX=196608   192K 比 224K/229K 更快（布局效应），且池更大
# KVD=auto     bf16 KV；fp8 KV 只在单请求 >100K 时净赚（长文 +197%，短文 −18%）
# KVOFF=16     超池部分落主机；20 会饿死 shm 广播，24 直接死锁
# MBT=2048     8192→2048 单流 +15%，prefill 只 −1.8%（hybrid GDN 的 chunk kernel 会挡 decode）
# RET=448      不设则 prefix cache 与投机互斥（缓存永不命中）
# SN=7         合法域 [5,7]；5 的 1c 只有 150，8c 631
