"""Build a structurally simple, render-checked SVG SFT dataset.

The source SVGs are valid but often contain gradients, filters, clip paths,
``<use>`` references, and oversized background rectangles.  Those constructs
are useful for authored SVGs but are unnecessarily difficult targets for a
small language model.  This script converts them to a shallow shapes-only
subset while preserving the visible logo as closely as practical.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import shutil
import statistics
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any


SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
ET.register_namespace("", SVG_NS)

SHAPE_TAGS = {"path", "circle", "ellipse", "rect", "polygon", "line"}
KEEP_TAGS = {"svg", "g", *SHAPE_TAGS}
REFERENCE_ATTRS = {"clip-path", "filter", "mask"}
FALLBACK_COLORS = (
    "#1f2937",
    "#2563eb",
    "#0f766e",
    "#d97706",
    "#dc2626",
    "#7c3aed",
)

SIMPLE_SYSTEM_PROMPT = """You are an expert logo designer working in clean, scalable vector graphics. Given a description of a logo's visual elements, output ONE complete SVG document for the logo.

Rules:
- Output ONLY one <svg>...</svg> element with xmlns="http://www.w3.org/2000/svg" and viewBox="0 0 256 256". No prose, markdown, or code fences.
- Use only <g>, <path>, <circle>, <ellipse>, <rect>, <polygon>, and <line> inside the SVG.
- Use solid fill and stroke colors only. Do not use <defs>, gradients, filters, masks, clip paths, <use>, external references, scripts, animation, text, or images.
- Keep the structure shallow, use at most 40 graphic elements, and finish with </svg>.
- Compose the requested logo centered within the 256 by 256 canvas using a small cohesive palette."""


def local_name(name: str) -> str:
    return name.rsplit("}", 1)[-1]


def stable_fallback(identifier: str) -> str:
    digest = hashlib.sha256(identifier.encode("utf-8")).digest()
    return FALLBACK_COLORS[int.from_bytes(digest[:4], "big") % len(FALLBACK_COLORS)]


def first_stop_color(gradient: ET.Element) -> str | None:
    for child in gradient.iter():
        if local_name(child.tag) != "stop":
            continue
        color = child.attrib.get("stop-color")
        if not color:
            match = re.search(r"(?:^|;)\s*stop-color\s*:\s*([^;]+)", child.attrib.get("style", ""), re.I)
            color = match.group(1).strip() if match else None
        if color and color.lower() not in {"none", "transparent"}:
            return color
    return None


def collect_definitions(root: ET.Element) -> tuple[dict[str, str], dict[str, ET.Element]]:
    gradients: dict[str, str] = {}
    definitions: dict[str, ET.Element] = {}
    for element in root.iter():
        identifier = element.attrib.get("id")
        if identifier:
            definitions[identifier] = element
        if local_name(element.tag) in {"linearGradient", "radialGradient"} and identifier:
            gradients[identifier] = first_stop_color(element) or stable_fallback(identifier)
    return gradients, definitions


def expand_use_elements(root: ET.Element, definitions: dict[str, ET.Element]) -> int:
    """Replace local <use> references with copied shapes/groups."""

    expanded = 0
    for parent in list(root.iter()):
        children = list(parent)
        for index, child in enumerate(children):
            if local_name(child.tag) != "use":
                continue
            href = child.attrib.get("href") or child.attrib.get(f"{{{XLINK_NS}}}href")
            target = definitions.get(href[1:]) if href and href.startswith("#") else None
            if target is None or local_name(target.tag) not in KEEP_TAGS - {"svg"}:
                parent.remove(child)
                continue

            clone = copy.deepcopy(target)
            clone.attrib.pop("id", None)
            x, y = child.attrib.get("x", "0"), child.attrib.get("y", "0")
            transforms = []
            if x not in {"", "0", "0.0"} or y not in {"", "0", "0.0"}:
                transforms.append(f"translate({x} {y})")
            if clone.attrib.get("transform"):
                transforms.append(clone.attrib["transform"])
            if child.attrib.get("transform"):
                transforms.append(child.attrib["transform"])
            if transforms:
                clone.attrib["transform"] = " ".join(transforms)
            for key, value in child.attrib.items():
                if local_name(key) not in {"href", "x", "y", "transform"}:
                    clone.attrib[key] = value
            parent.remove(child)
            parent.insert(index, clone)
            expanded += 1
    return expanded


def simplify_style(value: str, gradients: dict[str, str]) -> str:
    declarations = []
    for declaration in value.split(";"):
        if ":" not in declaration:
            continue
        key, val = (part.strip() for part in declaration.split(":", 1))
        if key.lower() in REFERENCE_ATTRS:
            continue
        if key.lower() in {"fill", "stroke"}:
            val = replace_paint_server(val, gradients)
        declarations.append(f"{key}:{val}")
    return ";".join(declarations)


def replace_paint_server(value: str, gradients: dict[str, str]) -> str:
    match = re.fullmatch(r"\s*url\(\s*#([^\s)]+)\s*\)\s*", value, re.I)
    if not match:
        return value
    identifier = match.group(1)
    return gradients.get(identifier, stable_fallback(identifier))


def normalize_number_text(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        number = float(match.group(0))
        if abs(number) < 0.0005:
            number = 0.0
        result = f"{number:.2f}".rstrip("0").rstrip(".")
        return "0" if result in {"-0", ""} else result

    return re.sub(r"[-+]?(?:\d+\.\d+|\d+\.)(?:e[-+]?\d+)?", replace, value, flags=re.I)


def normalize_background_rect(element: ET.Element) -> bool:
    if local_name(element.tag) != "rect":
        return False
    try:
        x = float(element.attrib.get("x", "0"))
        y = float(element.attrib.get("y", "0"))
        width = float(element.attrib.get("width", "0"))
        height = float(element.attrib.get("height", "0"))
    except ValueError:
        return False
    if x <= -256 and y <= -256 and width >= 768 and height >= 768:
        element.attrib.update({"x": "0", "y": "0", "width": "256", "height": "256"})
        return True
    return False


def remove_unsupported(root: ET.Element) -> Counter[str]:
    removed: Counter[str] = Counter()
    changed = True
    while changed:
        changed = False
        for parent in list(root.iter()):
            for child in list(parent):
                tag = local_name(child.tag)
                if tag not in KEEP_TAGS:
                    removed[tag] += 1
                    parent.remove(child)
                    changed = True
    return removed


def simplify_svg(svg_text: str) -> tuple[str, dict[str, Any]]:
    root = ET.fromstring(svg_text.strip())
    if local_name(root.tag) != "svg":
        raise ValueError("root element is not svg")

    gradients, definitions = collect_definitions(root)
    expanded_uses = expand_use_elements(root, definitions)
    removed = remove_unsupported(root)
    normalized_backgrounds = 0

    for element in root.iter():
        element.attrib.pop("id", None)
        for key in list(element.attrib):
            name = local_name(key).lower()
            value = element.attrib[key]
            if name in REFERENCE_ATTRS or name.startswith("on") or name == "href":
                element.attrib.pop(key, None)
            elif name in {"fill", "stroke"}:
                element.attrib[key] = replace_paint_server(value, gradients)
            elif name == "style":
                simplified = simplify_style(value, gradients)
                if simplified:
                    element.attrib[key] = simplified
                else:
                    element.attrib.pop(key, None)
            else:
                element.attrib[key] = normalize_number_text(value)
        normalized_backgrounds += int(normalize_background_rect(element))

    root.attrib["viewBox"] = "0 0 256 256"
    serialized = ET.tostring(root, encoding="unicode", short_empty_elements=True)
    serialized = re.sub(r">\s+<", "><", serialized).strip()
    ET.fromstring(serialized)  # Assert that serialization remains valid XML.
    graphics = sum(local_name(element.tag) in SHAPE_TAGS for element in root.iter())
    return serialized, {
        "gradients_replaced": len(gradients),
        "uses_expanded": expanded_uses,
        "normalized_backgrounds": normalized_backgrounds,
        "removed_tags": dict(removed),
        "graphics": graphics,
    }


def render_svg(svg: str, renderer: str) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory(prefix="svg-check-") as directory:
        source = Path(directory) / "input.svg"
        output = Path(directory) / "output.png"
        source.write_text(svg, encoding="utf-8")
        result = subprocess.run(
            [renderer, "-background", "none", str(source), str(output)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )
        ok = result.returncode == 0 and output.exists() and output.stat().st_size > 0
        return ok, result.stderr.strip()


def summarize(values: list[int]) -> dict[str, float | int]:
    ordered = sorted(values)
    if not ordered:
        return {"count": 0}

    def percentile(p: float) -> float:
        rank = (len(ordered) - 1) * p
        low = int(rank)
        high = min(low + 1, len(ordered) - 1)
        return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)

    return {
        "count": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "mean": round(statistics.fmean(values), 2),
        "p90": round(percentile(0.90), 2),
        "p95": round(percentile(0.95), 2),
        "max": max(values),
    }


def process_file(
    source: Path,
    destination: Path,
    renderer: str | None,
    *,
    max_answer_chars: int | None = None,
    max_graphics: int | None = None,
) -> dict[str, Any]:
    rows = []
    answer_before: list[int] = []
    answer_after: list[int] = []
    graphics: list[int] = []
    kept_answer_after: list[int] = []
    kept_graphics: list[int] = []
    totals: Counter[str] = Counter()
    removed_tags: Counter[str] = Counter()
    render_failures = []
    filtered = []
    source_count = 0

    for line_no, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        source_count += 1
        row = json.loads(line)
        for message in row["messages"]:
            if message.get("role") == "system":
                message["content"] = SIMPLE_SYSTEM_PROMPT
        assistant = next(message for message in row["messages"] if message["role"] == "assistant")
        original = assistant["content"]
        simplified, item = simplify_svg(original)
        assistant["content"] = simplified
        answer_before.append(len(original))
        answer_after.append(len(simplified))
        graphics.append(item["graphics"])
        totals.update({
            "gradients_replaced": item["gradients_replaced"],
            "uses_expanded": item["uses_expanded"],
            "normalized_backgrounds": item["normalized_backgrounds"],
        })
        removed_tags.update(item["removed_tags"])
        if renderer:
            ok, error = render_svg(simplified, renderer)
            if not ok:
                render_failures.append({"line": line_no, "error": error})
        reasons = []
        if max_answer_chars and len(simplified) > max_answer_chars:
            reasons.append(f"answer_chars>{max_answer_chars}")
        if max_graphics and item["graphics"] > max_graphics:
            reasons.append(f"graphics>{max_graphics}")
        if reasons:
            filtered.append(
                {
                    "line": line_no,
                    "answer_chars": len(simplified),
                    "graphics": item["graphics"],
                    "reasons": reasons,
                }
            )
            continue
        kept_answer_after.append(len(simplified))
        kept_graphics.append(item["graphics"])
        rows.append(row)

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "source": str(source),
        "destination": str(destination),
        "source_count": source_count,
        "count": len(rows),
        "filtered_count": len(filtered),
        "filtered": filtered,
        "answer_chars_before": summarize(answer_before),
        "answer_chars_after_all": summarize(answer_after),
        "graphics_after_all": summarize(graphics),
        "answer_chars_final": summarize(kept_answer_after),
        "graphics_final": summarize(kept_graphics),
        "operations": dict(totals),
        "removed_tags": dict(removed_tags),
        "xml_valid": source_count,
        "render_checked": source_count if renderer else 0,
        "render_failures": render_failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default="logo-detailed-prompt")
    parser.add_argument("--output-dir", default="logo-detailed-prompt-simple")
    parser.add_argument(
        "--renderer",
        default=shutil.which("convert"),
        help="SVG renderer executable used for validation; pass an empty value to disable.",
    )
    parser.add_argument(
        "--max-train-answer-chars",
        type=int,
        default=3200,
        help="Drop unusually long simplified training answers; 0 disables the limit.",
    )
    parser.add_argument(
        "--max-train-graphics",
        type=int,
        default=40,
        help="Drop training answers with too many graphic elements; 0 disables the limit.",
    )
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    renderer = args.renderer or None
    report = {
        "policy": {
            "gradient": "replace paint-server fill/stroke with first stop color",
            "use": "expand local shape/group references",
            "unsupported": "remove defs, filters, clipping, masks, animation, text, and style elements",
            "background": "normalize oversized full-canvas rectangles to 0 0 256 256",
            "coordinates": "round decimal attributes to two places",
        },
        "renderer": renderer,
        "train": process_file(
            source_dir / "train.jsonl",
            output_dir / "train.jsonl",
            renderer,
            max_answer_chars=args.max_train_answer_chars or None,
            max_graphics=args.max_train_graphics or None,
        ),
        "valid": process_file(source_dir / "valid.jsonl", output_dir / "valid.jsonl", renderer),
    }
    report["all_rendered"] = not (
        report["train"]["render_failures"] or report["valid"]["render_failures"]
    )
    (output_dir / "simplify_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["all_rendered"]:
        raise SystemExit("one or more simplified SVGs failed rendering")


if __name__ == "__main__":
    main()
