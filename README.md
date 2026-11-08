# club-3090x2 — 双 RTX 3090 跑 27B 稠密模型：单流 272 / 8 并发 774 tok/s

一台**纯消费级**双卡机（2× RTX 3090 24G，PCIe 4.0 **x8/x8**，无 NVLink）把 Qwen3.8-27B 级别的稠密模型
跑到 **单流 272 tok/s、8 并发 774 tok/s、prefill 3341 tok/s**，192K 上下文，GSM8K-500 精度 **0.952**。

没有用任何服务器平台、没有 NVLink、没有换卡——全部收益来自**量化格式 + 投机解码 + 通信/内核层的针对性改造**。

## 结果速览

| 项目 | 数值 | 说明 |
|---|---:|---|
| 单流（1c） | **272.0 tok/s** | 5 条中文短任务 × 256 tok，取中位（解码口径） |
| 4 并发 | **481.9 tok/s** | 聚合 |
| 8 并发 | **774.0 tok/s** | 聚合 |
| Prefill 8K / 32K / 96K | **3341 / 2890 / 2111 tok/s** | 随机串 prompt，3 次取中位 |
| GSM8K（500 题） | **0.952**（476/500） | greedy |
| HumanEval（164 题） | **0.951**（156/164） | `reasoning_effort=low`；不思考 0.927 |
| 上下文 | **196,608** | bf16 KV；KV 池 234,218 token + 16G 主机 offload |
| 显存占用 | 21.1 GB / 卡 | 权重 18G + KV + 草稿 |

## 三个关键决定（按收益排序）

### 1. 用 INT4 权重 + **INT8 激活**（W4A8），而不是常见的 W4A16

RTX 3090（sm86）**没有 FP8/FP4 张量核**，但有完整的 INT8 张量核（250 TOPS）。常见的 AutoRound/GPTQ
INT4 包只做权重压缩、激活仍是 BF16，等于把 int8 张量核白放着。

改成 W4A8 后 **prefill +19.6%**（2858 vs 2390 tok/s），代价是必须打补丁处理 AutoRound 导出的负 scale
（marlin 的 int8 group scale 会被当无符号读，不打补丁会输出乱码但跑分"正常"——非常隐蔽）。
补丁见 `docs/DEPLOY.md` 的「补丁清单」。

### 2. DFlash2 投机解码，草稿链深度 7

27B 稠密模型单步要读 ~18GB 权重，decode 天生受显存带宽限制。投机解码把每步产出从 1 token 拉到 3.1 token。
深度扫过 5/7（更小会显著掉吞吐，draft 的合法域实测是 [5,7]）：

| 草稿深度 | 1c | 8c |
|---:|---:|---:|
| 5 | 150.0 | 631.5 |
| **7** | **272.0** | **774.0** |

### 3. 双卡通信：P2P + 自研 all-reduce

- 两张卡各跑 Gen4 **x8**（消费级平台拆分的上限），默认无 P2P，all-reduce 走主机内存
- 打通 P2P 后实测 **13.19 GB/s**，prefill **+16~20%**
- 打开 vLLM 的 custom all-reduce（需要显存有余量才能初始化）：1c +9% / 4c +26% / 8c +12%

## 目录

```
README.md              本文件（结果与结论）
docs/DEPLOY.md         部署方案：环境、补丁、模型准备、启动脚本、参数表、坑位
docs/BENCHMARK.md      完整测试报告：口径定义、全部数据、被否掉的替代方案
scripts/serve.sh       启动脚本（参数化，直接可用）
scripts/bench_luz.py   吞吐测试（1c/4c/8c + prefill）
scripts/eval_gsm8k.py  精度测试（GSM8K，可指定题数）
```

## 模型

权重**没有改动**，只有量化，因此不在此仓库分发，直接用上游：

| 用途 | 下载地址 | 说明 |
|---|---|---|
| 主模型基座 | `bottlecapai/ThinkingCap-Qwen3.8-27B` | Qwen3.8-27B 混合架构（48 GDN + 16 full-attn，64 层，原生 262K）的推理微调 |
| 投机草稿 | `incoai/Qwen3.8-27B-DFlash2` | 公开的 DFlash2 草稿，需另做 W4A16（compressed-tensors）量化，或用现成 W4A16 包 |

主模型需要自己做一遍 **AutoRound INT4（W4A16 导出，运行时用 W4A8）**，配方见 `docs/DEPLOY.md`，
关键参数：`bits=4, group_size=128, sym=True, nsamples=512, seqlen=2048, iters=200`。

## 复现

```bash
# 1) 环境（见 docs/DEPLOY.md）
# 2) 量化基座模型 -> ~/models/qwen3.8-27b-autoround-int4
# 3) 启动
AI=~/ai MODEL=~/models/qwen3.8-27b-autoround-int4 \
DRAFT=~/models/dflash2-w4a16 CTX=196608 SEQ=16 MBT=2048 SN=7 KVOFF=16 \
bash scripts/serve.sh

# 4) 测速
python scripts/bench_luz.py --port 8000 --model q27

# 5) 测精度
python scripts/eval_gsm8k.py --port 8000 --model q27 --n 500 --data gsm8k_test.jsonl
```

## 硬件与许可

- 测试机：2× RTX 3090（48G 合计，PCIe 4.0 x8/x8，P2P 已开）、消费级 16 核 CPU、64G DDR5、单块 NVMe
- 模型许可：ThinkingCap 为 **PolyForm Small Business 1.0.0**（商用前请自行确认）；DFlash2 为 Apache-2.0
- 本仓库脚本：MIT

> 数据均为本机实测（2026-11），口径写在 `docs/BENCHMARK.md`，与他人口径不可直接横比。
> 同一配置重复测的波动：1c 在 **238–272** tok/s、8c 在 **748–774** tok/s（受 prefix cache 命中与投机突发影响），
> prefill 基本不动（±1%）。上表取多次测量中的较高值，复现时落在区间内即正常。
