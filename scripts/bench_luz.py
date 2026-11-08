#!/usr/bin/env python3
"""luz 口径吞吐测试：1c / 4c / 8c + prefill 8K/32K/96K。

    python bench_luz.py --port 8000 --model q27 [--key sk-xxx] [--conc 4,8]

口径说明（与 docs/BENCHMARK.md 一致）：
- 1c   : 5 条中文短任务各 256 tok，temperature=0，取中位；只算首 token→末 token（不含 prefill）
- Nc   : 并发 N 路，60 字符随机 prompt + "写代码"，各 128 tok，聚合 = 总生成 / 墙钟
- PF-xK: 随机串 prompt，max_tokens=8，3 次取中位，= prompt_tokens / TTFT
"""
import argparse
import json
import random
import statistics
import string
import threading
import time
import urllib.request

TASKS = ["用Python写LRU缓存类", "实现线程安全队列", "写SQL分页查询", "解释三次握手", "实现快速排序"]


def call(url, key, model, prompt, max_tokens, timeout=1800):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(url, data=body, headers=headers)

    t0 = time.time()
    first = last = None
    ntok = ptok = 0
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for line in resp:
            if not line.startswith(b"data: ") or b"[DONE]" in line:
                continue
            d = json.loads(line[6:])
            if d.get("usage"):
                ntok = d["usage"].get("completion_tokens", 0)
                ptok = d["usage"].get("prompt_tokens", 0)
            ch = d.get("choices")
            if ch:
                delta = ch[0].get("delta") or {}
                if delta.get("content") or delta.get("reasoning") or delta.get("reasoning_content"):
                    if first is None:
                        first = time.time()
                    last = time.time()
    if first is None:
        return ptok, ntok, -1, -1
    return ptok, ntok, first - t0, max(last - first, 1e-3)


def filler(lines):
    return "".join("".join(random.choices(string.ascii_lowercase + string.digits, k=60)) + "\n"
                   for _ in range(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default="q27")
    ap.add_argument("--key", default="")
    ap.add_argument("--conc", default="4,8", help="并发档，逗号分隔")
    args = ap.parse_args()
    url = f"http://{args.host}:{args.port}/v1/chat/completions"

    r = []
    for t in TASKS:
        _, nt, ttft, dec = call(url, args.key, args.model, t, 256)
        if ttft >= 0:
            r.append(nt / dec)
    print("1c: %.1f t/s" % statistics.median(r) if r else "1c: FAIL", flush=True)

    for conc in [int(x) for x in args.conc.split(",")]:
        out = []
        lock = threading.Lock()

        def worker():
            p = "".join(random.choices(string.ascii_lowercase, k=60)) + " 写代码"
            _, nt, ttft, dec = call(url, args.key, args.model, p, 128)
            if ttft >= 0:
                with lock:
                    out.append(nt)

        threads = [threading.Thread(target=worker) for _ in range(conc)]
        t0 = time.time()
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        wall = time.time() - t0
        print("%dc: ok=%d agg=%.1f" % (conc, len(out), sum(out) / wall) if out
              else "%dc: FAIL" % conc, flush=True)

    for label, lines in (("8K", 300), ("32K", 1200), ("96K", 3600)):
        pts = []
        for _ in range(3):
            pt, _, ttft, _ = call(url, args.key, args.model, filler(lines) + "\nOK", 8)
            if ttft > 0:
                pts.append(pt / ttft)
            time.sleep(0.5)
        print("PF-%s: %.0f tok/s" % (label, statistics.median(pts)) if pts
              else "PF-%s: FAIL" % label, flush=True)


if __name__ == "__main__":
    main()
