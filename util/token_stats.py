"""Token length statistics for SVG-LoRA jsonl datasets."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="logo-detailed-prompt/train.jsonl", help="JSONL dataset path.")
    parser.add_argument("--model", default="./gemma3-270m", help="Tokenizer model id or local path.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many shortest and longest samples to print.",
    )
    parser.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Tokenize raw message contents joined by newlines instead of applying the tokenizer chat template.",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    rows = load_rows(Path(args.data))
    stats = collect_lengths(tokenizer, rows, use_chat_template=not args.no_chat_template)
    print_report(Path(args.data), stats, args.top_k)


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_line_no"] = line_no
            rows.append(row)
    return rows


def collect_lengths(tokenizer, rows: list[dict[str, Any]], *, use_chat_template: bool) -> list[dict[str, Any]]:
    stats = []
    for idx, row in enumerate(rows):
        messages = row.get("messages")
        if not isinstance(messages, list):
            raise ValueError(f"line {row.get('_line_no', idx + 1)} does not contain a messages list")

        if use_chat_template and getattr(tokenizer, "chat_template", None):
            total_token_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        else:
            text = "\n".join(f"{m.get('role', '')}: {m.get('content', '')}" for m in messages)
            total_token_ids = tokenizer(text, add_special_tokens=True)["input_ids"]

        prompt = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
        answer = next((m.get("content", "") for m in messages if m.get("role") == "assistant"), "")
        prompt_token_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        answer_token_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
        stats.append(
            {
                "index": idx,
                "line_no": row.get("_line_no", idx + 1),
                "total_tokens": len(total_token_ids),
                "prompt_tokens": len(prompt_token_ids),
                "answer_tokens": len(answer_token_ids),
                "prompt_chars": len(prompt),
                "answer_chars": len(answer),
            }
        )
    return stats


def percentile(values: list[int], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def print_report(path: Path, stats: list[dict[str, Any]], top_k: int) -> None:
    lengths = [item["total_tokens"] for item in stats]
    if not lengths:
        print(f"{path}: no samples")
        return

    print(f"file: {path}")
    print(f"count: {len(lengths)}")
    print_token_summary("total_tokens", [item["total_tokens"] for item in stats])
    print_token_summary("prompt_tokens", [item["prompt_tokens"] for item in stats])
    print_token_summary("answer_tokens", [item["answer_tokens"] for item in stats])

    top_k = max(0, top_k)
    if top_k:
        print()
        print(f"shortest answer {top_k}:")
        for item in sorted(stats, key=lambda x: x["answer_tokens"])[:top_k]:
            print(format_item(item))

        print()
        print(f"longest answer {top_k}:")
        for item in sorted(stats, key=lambda x: x["answer_tokens"], reverse=True)[:top_k]:
            print(format_item(item))


def print_token_summary(name: str, lengths: list[int]) -> None:
    print(f"{name}:")
    print(f"  min: {min(lengths)}")
    print(f"  max: {max(lengths)}")
    print(f"  avg: {statistics.fmean(lengths):.2f}")
    print(f"  median: {statistics.median(lengths):.2f}")
    print(f"  p90: {percentile(lengths, 0.90):.2f}")
    print(f"  p95: {percentile(lengths, 0.95):.2f}")


def format_item(item: dict[str, Any]) -> str:
    return (
        f"index={item['index']} line={item['line_no']} "
        f"total_tokens={item['total_tokens']} prompt_tokens={item['prompt_tokens']} "
        f"answer_tokens={item['answer_tokens']} "
        f"prompt_chars={item['prompt_chars']} answer_chars={item['answer_chars']}"
    )


if __name__ == "__main__":
    main()
