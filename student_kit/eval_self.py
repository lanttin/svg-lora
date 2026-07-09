"""Self-evaluation for base vs LoRA-adapted SVG generation with ms-swift."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm
from swift import InferRequest, RequestConfig, TransformersEngine

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from student_kit.reward import score_svg


SYSTEM_PROMPT = """You are an expert logo designer working in clean, scalable vector graphics. Given a description of a logo's visual elements, output ONE complete SVG document for the logo.

Rules:
- Output ONLY the SVG: a single <svg ...>...</svg> element with an xmlns and viewBox="0 0 256 256". No prose, no markdown, no code fences.
- Compose centered, content roughly within 16..240. Use a small cohesive palette.
- Put gradients/filters in <defs>; use vector primitives only (<path>, <circle>, <ellipse>, <rect>, <polygon>, <line>, <g>). No <image>, external refs, or scripts.
- Draw exactly what the description specifies."""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Base model id or local path.")
    parser.add_argument("--adapter", default=None, help="LoRA adapter path. If omitted, only base is evaluated.")
    parser.add_argument("--valid", default="logo-detailed-prompt/valid.jsonl")
    parser.add_argument("--output", default="results.json")
    parser.add_argument("--max-new-tokens", type=int, default=1400)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    examples = load_examples(Path(args.valid))
    if args.limit:
        examples = examples[: args.limit]

    results: dict[str, Any] = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "valid_file": args.valid,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "runs": {},
    }

    base_engine = TransformersEngine(args.model)
    results["runs"]["base"] = evaluate_engine(
        base_engine,
        examples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    if args.adapter:
        adapter_engine = TransformersEngine(args.model, adapters=[args.adapter])
        results["runs"]["adapter"] = evaluate_engine(
            adapter_engine,
            examples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )

    Path(args.output).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print_summary(results)


def load_examples(path: Path) -> list[dict[str, str]]:
    examples = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            messages = row["messages"]
            user = next(m["content"] for m in messages if m["role"] == "user")
            assistant = next(m["content"] for m in messages if m["role"] == "assistant")
            system = next((m["content"] for m in messages if m["role"] == "system"), SYSTEM_PROMPT)
            examples.append({"system": system, "prompt": user, "target": assistant})
    return examples


def evaluate_engine(
    engine: TransformersEngine,
    examples: list[dict[str, str]],
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> dict[str, Any]:
    items = []
    for idx, ex in enumerate(tqdm(examples, desc="eval")):
        output = generate_svg(
            engine,
            ex["system"],
            ex["prompt"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        scored = score_svg(output, ex["prompt"])
        items.append(
            {
                "index": idx,
                "prompt": ex["prompt"],
                "target": ex["target"],
                "prediction": output,
                "reward": scored,
            }
        )
    scores = [item["reward"]["score"] for item in items]
    return {
        "mean_reward": sum(scores) / len(scores) if scores else 0.0,
        "count": len(items),
        "items": items,
    }


def generate_svg(
    engine: TransformersEngine,
    system: str,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    request = InferRequest(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
    )
    config = RequestConfig(
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    response = engine.infer([request], config)[0]
    return response.choices[0].message.content.strip()


def print_summary(results: dict[str, Any]) -> None:
    for name, run in results["runs"].items():
        print(f"{name}: mean_reward={run['mean_reward']:.4f}, count={run['count']}")
    if "base" in results["runs"] and "adapter" in results["runs"]:
        delta = results["runs"]["adapter"]["mean_reward"] - results["runs"]["base"]["mean_reward"]
        print(f"delta(adapter-base): {delta:+.4f}")


if __name__ == "__main__":
    main()
