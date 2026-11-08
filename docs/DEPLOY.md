# 部署方案

目标：在 **2× RTX 3090（48G，PCIe x8/x8，无 NVLink）** 上把 27B 稠密模型跑到 192K 上下文、
单流 270 tok/s、8 并发 770 tok/s。

---

## 1. 硬件

| 项 | 要求 | 备注 |
|---|---|---|
| GPU | 2× RTX 3090 24G（sm86） | 24G×2 是硬门槛：18G 权重 + KV + 草稿 |
| PCIe | 每卡 ≥ x8（Gen4） | x8/x8 是消费级主板 16 lane 拆分的上限 |
| CPU | 任意 16 核以上 | 只做调度，不参与计算 |
| 内存 | ≥ 64G | 其中 16G 给 KV offload |
| 磁盘 | ≥ 300G 可用 | 权重 18G + 草稿 1.2G + 编译缓存 |

> **x8 是不是瓶颈？** 不是。decode 受显存带宽限制，all-reduce 每步只有 ~10MB；prefill 的 all-reduce 占比
> 约 50%，打通 P2P 后这部分已接近 PCIe x8 的理论值。

## 2. 软件栈

| 组件 | 版本 | 说明 |
|---|---|---|
| 驱动 | 595.x | 需要支持 P2P 的自编译模块（见 §4） |
| CUDA | 13.x | 用 venv 内的 `nvidia/cu13`，不要用系统盘上的残留目录 |
| vLLM | 0.32.0 | 带下面「补丁清单」中的改动 |
| Python | 3.12 | venv |

### 补丁清单（缺一不可）

1. **W4A8 负 scale 修正**：AutoRound 导出的 group scale 约一半为负，marlin 的 int8 路径按无符号读 →
   输出乱码（但吞吐指标完全正常，极易误判）。补丁把符号折进 int4 code。
2. **INT8 层选择**：`lm_head` 与 `mtp` 必须排除在 int8 激活之外（`VLLM_MARLIN_INT8_EXCLUDE_RE`），
   否则掉精度。
3. **自研 int8 lm_head**：词表投影是 248320×5120 的 BF16（2.54G），每步全量读，占单步字节数 13%。
   首次使用时按列量化成 int8 并用 Triton int8 GEMM 跑，省 0.64G/卡，单流 +15~50%。
   （`BM` 必须跟 batch 走：M=8 时用 BM=64 会白算 8 倍，实测 875 vs 424 GB/s。）
4. **custom all-reduce 打开**：需要显存留够余量才能初始化；打不开时 1c −9% / 4c −26%。

## 3. 模型准备

### 3.1 基座

```bash
hf download bottlecapai/ThinkingCap-Qwen3.8-27B --local-dir ~/models/thinkingcap-bf16
```

### 3.2 AutoRound INT4（W4A16 导出，运行时 W4A8）

54G 的 BF16 底模可以在 48G 双卡上量化（逐层换入换出）：

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m auto_round quantize \
  --model_name ~/models/thinkingcap-bf16 \
  --bits 4 --group_size 128 \
  --format auto_round \
  --output_dir ~/models/qwen3.8-27b-autoround-int4 \
  --device_map 0,1 --low_gpu_mem_usage \
  --nsamples 512 --seqlen 2048 --batch_size 1 --iters 200 \
  --enable_alg_ext --enable_torch_compile
```

- 校准集用默认的 `NeelNanda/pile-10k`（国内先 `HF_ENDPOINT=<mirror> hf download` 预取）
- `--sym` 不是合法参数（对称是默认，非对称才写 `--asym`）
- 27B 约 30–60 分钟，产出 ~18G

### 3.3 投机草稿

用 `incoai/Qwen3.8-27B-DFlash2` 的 **W4A16（compressed-tensors, group 128）** 版本，1.2G。
bf16 原版也能跑但**更慢**（实测 1c 160 vs 177、8 并发 557 vs 663，且多占 2.4G 显存），不建议。

## 4. PCIe P2P（可选但强烈建议）

两张 3090 在消费级主板上默认**不能** P2P，all-reduce 走主机内存。打通步骤：

1. 两张卡刷成**同型号**的 ReBAR vBIOS（不同 vBIOS 的组合会被驱动拒绝）
2. 用与驱动同版本的内核源码编译带 P2P 支持的 `nvidia` 模块
3. 内核启动参数加 `amd_iommu=on iommu=pt pcie_acs_override=downstream,multifunction`
4. 验证：`nvidia-smi topo -p2p r` 全 OK；实测 **13.19 GB/s**

收益：prefill **+16~20%**、4 并发 **+14%**、单流 +5%。

## 5. 启动

```bash
AI=~/ai \
MODEL=~/models/qwen3.8-27b-autoround-int4 \
DRAFT=~/models/dflash2-w4a16 \
CTX=196608 SEQ=16 MBT=2048 SN=7 KVOFF=16 RET=448 \
API_KEY=sk-your-key \
bash scripts/serve.sh
```

### 关键参数

| 参数 | 值 | 为什么 |
|---|---|---|
| `--tensor-parallel-size` | 2 | 单卡装不下 18G 权重 + 192K KV |
| `--max-model-len` | **196608** | 192K 比 224K/229K **更快**（同配置 229K→192K 时散文 +13%），不是显存问题而是布局 |
| `--kv-cache-dtype` | auto（bf16） | sm86 无 FP8 张量核，fp8 KV 短文净亏 18%，只在 >100K 单请求时划算 |
| `--kv-offloading-size` | **16**（G） | 超出 GPU 池的部分落主机内存；**20 会饿死 shm 广播，24 直接死锁** |
| `--max-num-batched-tokens` | **2048** | 8192→2048 使单流 +15%（hybrid GDN 的 chunk kernel 会阻塞 decode），prefill 只 −1.8% |
| `--prefix-cache-retention-interval` | **448** | 必须等于 `tokens_per_chunk` 的整数倍；不设则 prefix cache 与投机互斥（缓存永不命中） |
| `--speculative-config` | dflash，`num_speculative_tokens=7` | 合法域 [5,7]，7 最优 |
| `VLLM_MARLIN_INPUT_DTYPE` | int8 | 这就是 W4A8 的开关 |
| `VLLM_USE_FLASHINFER_SAMPLER` | 0 | FlashInfer 采样器在 sm86 上会挂 |

## 6. 坑位清单

| 现象 | 原因 | 解法 |
|---|---|---|
| 输出乱码但吞吐正常 | AutoRound 负 scale 被当无符号读 | 打 W4A8 负 scale 补丁 |
| `prefix_cache_retention_interval (448) must be a multiple of tokens_per_chunk (N)` | N 随投机配置变 | 报错里有实际值，按它改；实测 SN=7→448、SN=5→432、SN=4→432、MTP n=1→416、无投机→400 |
| 启动报 `Free memory ... less than desired` | 上一个服务的孤儿 worker 占卡 | `nvidia-smi --query-compute-apps=pid,used_memory` 找出真 PID 再 `kill -9`，确认显存回 15MiB |
| `No available shared memory broadcast block found in 60 seconds` | `/dev/shm` 残留 或 KV offload 开太大 | 清 `/dev/shm/psm_*`、`sem.mp-*`、`vllm_offload_*`；`--kv-offloading-size` 不超过 16 |
| 换 spec / 换量化后首次启动极慢（5–10 分钟） | torch.compile 全量重编译（AOT 缓存按配置分 key） | 正常；别在实验时反复重启，批量改完一次起 |
| 关思考不生效 | `/no_think` 在 chat template 里无效 | 用 `chat_template_kwargs: {"enable_thinking": false}` |
| `medium` 档反而比 `xhigh` 更啰嗦 | 模板里 medium 不加任何指令 | 用 `low` 或 `xhigh`，别用 medium |

## 7. 运维

- **显存预算**：`--gpu-memory-utilization 0.93` 起点；改 KV 相关参数后先看启动日志的
  `GPU KV cache size: N tokens`
- **服务重启**：杀主进程后务必确认显存回 15MiB 再起
- **编译缓存**：`~/.cache/vllm/torch_compile_cache` 会长到几十 G，磁盘紧张时可清（代价=下次重编译）
