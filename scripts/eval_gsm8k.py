#!/usr/bin/env python3
"""GSM8K 精度测试（默认 500 题）。

    python eval_gsm8k.py --port 8000 --model q27 --data gsm8k_test.jsonl --n 500

判分要点：模型常输出 `#### 18.`（带句号）或 `1,234`（带逗号），必须做数字归一化，
否则会系统性判错（历史数据曾因此整体偏低）。

数据格式：每行一个 JSON，含 `question` 与 `answer`（answer 里 `#### <number>` 结尾）。
"""
import argparse
import json
import re
import urllib.request


def norm(x):
    x = x.replace(",", "").strip().rstrip(".").strip()
    try:
        f = float(x)
        return str(int(f)) if f == int(f) else str(f)
    except Exception:
        return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default="q27")
    ap.add_argument("--key", default="")
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--thinking", default="off", choices=["off", "low", "medium", "xhigh"])
    args = ap.parse_args()

    url = f"http://{args.host}:{args.port}/v1/chat/completions"
    ctk = ({"enable_thinking": False} if args.thinking == "off"
           else {"reasoning_effort": args.thinking})

    qs = [json.loads(l) for l in open(args.data)][: args.n]
    ok = trunc = 0
    for i, q in enumerate(qs):
        prompt = q["question"] + "\nSolve step by step. End with '#### <number>'."
        body = json.dumps({
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": args.max_tokens,
            "temperature": 0,
            "chat_template_kwargs": ctk,
        }).encode()
        headers = {"Content-Type": "application/json"}
        if args.key:
            headers["Authorization"] = f"Bearer {args.key}"
        try:
            req = urllib.request.Request(url, data=body, headers=headers)
            with urllib.request.urlopen(req, timeout=900) as resp:
                d = json.load(resp)
            choice = d["choices"][0]
            if choice.get("finish_reason") == "length":
                trunc += 1
            m = choice["message"]
            txt = (m.get("content") or "") + (m.get("reasoning") or "") + (m.get("reasoning_content") or "")
            nums = re.findall(r"####\s*(-?[\d\.,]+)", txt)
            gold = norm(str(q["answer"]).split("####")[-1].split("\n")[0])
            if nums and norm(nums[-1]) == gold:
                ok += 1
        except Exception:
            pass
        if (i + 1) % 20 == 0:
            print("  %d/%d -> %d ok (%.1f%%)" % (i + 1, len(qs), ok, 100 * ok / (i + 1)), flush=True)

    print("GSM8K-%d [%s effort=%s]: %d/%d = %.3f (truncated %d)"
          % (len(qs), args.model, args.thinking, ok, len(qs), ok / max(len(qs), 1), trunc))


if __name__ == "__main__":
    main()
