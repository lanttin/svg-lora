"""Self-evaluation for base vs LoRA-adapted SVG generation with ms-swift."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm
from swift import InferRequest, RequestConfig, TransformersEngine
from transformers import AutoTokenizer, set_seed

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from student_kit.reward import score_svg


SYSTEM_PROMPT = """You are an expert logo designer working in clean, scalable vector graphics. Given a description of a logo's visual elements, output ONE complete SVG document for the logo.

Rules:
- Output ONLY one <svg>...</svg> element with xmlns="http://www.w3.org/2000/svg" and viewBox="0 0 256 256". No prose, markdown, or code fences.
- Use only <g>, <path>, <circle>, <ellipse>, <rect>, <polygon>, and <line> inside the SVG.
- Use solid fill and stroke colors only. Do not use <defs>, gradients, filters, masks, clip paths, <use>, external references, scripts, animation, text, or images.
- Keep the structure shallow, use at most 40 graphic elements, and finish with </svg>.
- Compose the requested logo centered within the 256 by 256 canvas using a small cohesive palette."""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Base model id or local path.")
    parser.add_argument("--adapter", default=None, help="LoRA adapter path. If omitted, only base is evaluated.")
    parser.add_argument("--valid", default="logo-detailed-prompt-simple/valid.jsonl")
    parser.add_argument("--output", default="results.json")
    parser.add_argument("--max-new-tokens", type=int, default=1600)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
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
        "seed": args.seed,
        "runs": {},
    }

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    base_engine = TransformersEngine(args.model)
    set_seed(args.seed)
    results["runs"]["base"] = evaluate_engine(
        base_engine,
        tokenizer,
        examples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    if args.adapter:
        adapter_engine = TransformersEngine(args.model, adapters=[args.adapter])
        # Give base and adapter the same sampling stream for a fair comparison.
        set_seed(args.seed)
        results["runs"]["adapter"] = evaluate_engine(
            adapter_engine,
            tokenizer,
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
    tokenizer,
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
        processed = process_generation(output)
        scored = score_svg(output, ex["prompt"])
        prediction_tokens = count_tokens(tokenizer, output)
        extracted_svg_tokens = count_tokens(tokenizer, processed["svg"]) if processed["svg"] else 0
        items.append(
            {
                "index": idx,
                "prompt": ex["prompt"],
                "target": ex["target"],
                "prediction": output,
                "extracted_svg": processed["svg"],
                "closed_svg_found": processed["closed_svg_found"],
                "extra_text_before_svg": processed["extra_before"],
                "extra_text_after_svg": processed["extra_after"],
                "prediction_chars": len(output),
                "prediction_tokens": prediction_tokens,
                "extracted_svg_tokens": extracted_svg_tokens,
                "hit_max_new_tokens": prediction_tokens >= max_new_tokens - 2,
                "reward": scored,
            }
        )
    scores = [item["reward"]["score"] for item in items]
    token_counts = [item["prediction_tokens"] for item in items]
    return {
        "mean_reward": sum(scores) / len(scores) if scores else 0.0,
        "count": len(items),
        "output_tokens": {
            "mean": sum(token_counts) / len(token_counts) if token_counts else 0.0,
            "min": min(token_counts) if token_counts else 0,
            "max": max(token_counts) if token_counts else 0,
            "hit_max_new_tokens_count": sum(item["hit_max_new_tokens"] for item in items),
        },
        "items": items,
    }


def count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def process_generation(text: str) -> dict[str, Any]:
    match = re.search(r"<svg\b[\s\S]*?</svg>", text or "", flags=re.IGNORECASE)
    if not match:
        open_match = re.search(r"<svg\b", text or "", flags=re.IGNORECASE)
        return {
            "svg": "",
            "closed_svg_found": False,
            "extra_before": bool(open_match and text[:open_match.start()].strip()),
            "extra_after": False,
        }
    return {
        "svg": match.group(0),
        "closed_svg_found": True,
        "extra_before": bool(text[:match.start()].strip()),
        "extra_after": bool(text[match.end():].strip()),
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
        output_tokens = run.get("output_tokens", {})
        print(
            f"{name}: mean_reward={run['mean_reward']:.4f}, count={run['count']}, "
            f"output_tokens(mean/min/max)="
            f"{output_tokens.get('mean', 0):.1f}/"
            f"{output_tokens.get('min', 0)}/"
            f"{output_tokens.get('max', 0)}, "
            f"hit_max_new_tokens={output_tokens.get('hit_max_new_tokens_count', 0)}/{run['count']}"
        )
    if "base" in results["runs"] and "adapter" in results["runs"]:
        delta = results["runs"]["adapter"]["mean_reward"] - results["runs"]["base"]["mean_reward"]
        print(f"delta(adapter-base): {delta:+.4f}")


if __name__ == "__main__":
    main()
