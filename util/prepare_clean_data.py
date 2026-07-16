"""Create cleaned SVG-LoRA datasets by filtering long training samples."""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default="logo-detailed-prompt")
    parser.add_argument("--output-dir", default="logo-detailed-prompt-cleaned")
    parser.add_argument("--model", default="./gemma3-270m")
    parser.add_argument("--max-train-tokens", type=int, default=2048)
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    train_stats = filter_train(
        tokenizer,
        source_dir / "train.jsonl",
        output_dir / "train.jsonl",
        max_tokens=args.max_train_tokens,
    )
    shutil.copyfile(source_dir / "valid.jsonl", output_dir / "valid.jsonl")

    report = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "model": args.model,
        "max_train_tokens": args.max_train_tokens,
        "train": train_stats,
        "valid": {"policy": "copied unchanged"},
    }
    (output_dir / "filter_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def filter_train(tokenizer, input_path: Path, output_path: Path, *, max_tokens: int) -> dict[str, Any]:
    kept = []
    removed = []
    with input_path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            token_count = total_tokens(tokenizer, row["messages"])
            item = {"line_no": line_no, "total_tokens": token_count}
            if token_count <= max_tokens:
                kept.append((row, item))
            else:
                removed.append(item)

    with output_path.open("w", encoding="utf-8") as f:
        for row, _ in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    kept_lengths = [item["total_tokens"] for _, item in kept]
    removed_lengths = [item["total_tokens"] for item in removed]
    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "input_count": len(kept) + len(removed),
        "kept_count": len(kept),
        "removed_count": len(removed),
        "kept_total_tokens": summarize(kept_lengths),
        "removed_total_tokens": summarize(removed_lengths),
        "removed": removed,
    }


def total_tokens(tokenizer, messages: list[dict[str, str]]) -> int:
    if getattr(tokenizer, "chat_template", None):
        return len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False))
    text = "\n".join(f"{m.get('role', '')}: {m.get('content', '')}" for m in messages)
    return len(tokenizer(text, add_special_tokens=True)["input_ids"])


def summarize(lengths: list[int]) -> dict[str, float | int]:
    if not lengths:
        return {"count": 0, "min": 0, "max": 0, "avg": 0.0, "median": 0.0}
    return {
        "count": len(lengths),
        "min": min(lengths),
        "max": max(lengths),
        "avg": statistics.fmean(lengths),
        "median": statistics.median(lengths),
    }


if __name__ == "__main__":
    main()
