"""Select the lowest-eval-loss checkpoint from the latest training run."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from pathlib import Path
from typing import Any


def checkpoint_step(path: Path) -> int:
    match = re.fullmatch(r"checkpoint-(\d+)", path.name)
    return int(match.group(1)) if match else -1


def valid_checkpoint(path: Path) -> bool:
    return (
        path.is_dir()
        and checkpoint_step(path) >= 0
        and (path / "adapter_config.json").is_file()
        and (path / "adapter_model.safetensors").is_file()
        and (path / "trainer_state.json").is_file()
    )


def discover_latest_run(output_root: Path) -> tuple[Path, list[Path]]:
    checkpoints = [path for path in output_root.rglob("checkpoint-*") if valid_checkpoint(path)]
    if not checkpoints:
        raise FileNotFoundError(f"no complete LoRA checkpoints found under {output_root}")

    by_run: dict[Path, list[Path]] = {}
    for checkpoint in checkpoints:
        by_run.setdefault(checkpoint.parent, []).append(checkpoint)
    run_dir = max(
        by_run,
        key=lambda directory: max(path.stat().st_mtime for path in by_run[directory]),
    )
    return run_dir, sorted(by_run[run_dir], key=checkpoint_step)


def eval_loss_at_step(state: dict[str, Any], step: int) -> float | None:
    losses = [
        entry["eval_loss"]
        for entry in state.get("log_history", [])
        if entry.get("step") == step and isinstance(entry.get("eval_loss"), (int, float))
    ]
    return float(losses[-1]) if losses else None


def read_checkpoint_loss(checkpoint: Path) -> float | None:
    state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    return eval_loss_at_step(state, checkpoint_step(checkpoint))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="output/swift-svg-lora")
    parser.add_argument("--run-dir", default=None, help="Explicit run directory; defaults to the latest run.")
    parser.add_argument("--adapter-output", default="adapter")
    parser.add_argument("--report", default="adapter_selection.json")
    args = parser.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir)
        checkpoints = sorted(
            [path for path in run_dir.glob("checkpoint-*") if valid_checkpoint(path)],
            key=checkpoint_step,
        )
    else:
        run_dir, checkpoints = discover_latest_run(Path(args.output_root))
    if not checkpoints:
        raise FileNotFoundError(f"no complete checkpoints found in {run_dir}")

    evaluations = []
    for checkpoint in checkpoints:
        loss = read_checkpoint_loss(checkpoint)
        evaluations.append(
            {
                "checkpoint": str(checkpoint),
                "step": checkpoint_step(checkpoint),
                "eval_loss": loss,
                "eligible": loss is not None,
            }
        )

    eligible = [item for item in evaluations if item["eligible"]]
    if not eligible:
        raise RuntimeError(
            "none of the existing checkpoints has eval_loss at its checkpoint step; "
            "ensure eval_steps and save_steps are equal"
        )
    best = min(eligible, key=lambda item: (item["eval_loss"], item["step"]))
    best_checkpoint = Path(best["checkpoint"])

    adapter_output = Path(args.adapter_output)
    adapter_output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_checkpoint / "adapter_config.json", adapter_output / "adapter_config.json")
    shutil.copy2(
        best_checkpoint / "adapter_model.safetensors",
        adapter_output / "adapter_model.safetensors",
    )

    report = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_dir": str(run_dir),
        "selection_rule": "minimum eval_loss among existing checkpoints in the latest run",
        "evaluations": evaluations,
        "selected": best,
        "adapter_output": str(adapter_output),
    }
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"selected": best, "adapter_output": str(adapter_output)}, indent=2))


if __name__ == "__main__":
    main()
