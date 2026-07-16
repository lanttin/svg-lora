"""Programmatic reward for detailed-prompt -> SVG logo generation.

The reward is intentionally conservative: it rewards valid, bounded, simple SVG
logos before trying to judge semantic fidelity. This matches the assignment goal
for Gemma 3 270M: stable valid SVG is already meaningful progress.
"""

from __future__ import annotations

import math
import re
import statistics
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any


ALLOWED_TAGS = {
    "svg",
    "defs",
    "g",
    "path",
    "circle",
    "ellipse",
    "rect",
    "polygon",
    "line",
    "linearGradient",
    "radialGradient",
    "stop",
}

FORBIDDEN_TAGS = {
    "script",
    "image",
    "foreignObject",
    "iframe",
    "object",
    "embed",
    "video",
    "audio",
    "style",
    "text",
    "use",
}

GRAPHIC_TAGS = {"path", "circle", "ellipse", "rect", "polygon", "line"}

COLOR_WORDS = {
    "black": "#000000",
    "white": "#ffffff",
    "navy": "#1b3a5c",
    "blue": "#3366cc",
    "teal": "#008080",
    "green": "#2e8b57",
    "gold": "#d4af37",
    "golden": "#d4af37",
    "orange": "#f2a93b",
    "yellow": "#ffd54f",
    "red": "#d64545",
    "pink": "#e78ab3",
    "purple": "#7b4bb7",
    "brown": "#6b4226",
    "walnut": "#6b4226",
    "tan": "#d2a86a",
    "cream": "#fbf3e3",
    "gray": "#808080",
    "grey": "#808080",
}

SHAPE_HINTS = {
    "circle": "circle",
    "circular": "circle",
    "disc": "circle",
    "disk": "circle",
    "badge": "circle",
    "ring": "circle",
    "dot": "circle",
    "dots": "circle",
    "ellipse": "ellipse",
    "oval": "ellipse",
    "rect": "rect",
    "rectangle": "rect",
    "square": "rect",
    "line": "line",
    "lines": "line",
    "bar": "line",
    "bars": "line",
    "polygon": "polygon",
    "triangle": "polygon",
    "arrow": "path",
    "leaf": "path",
    "sprout": "path",
    "swirl": "path",
    "curve": "path",
    "wave": "path",
    "shield": "path",
    "star": "polygon",
    "column": "rect",
}


@dataclass
class RewardResult:
    score: float
    components: dict[str, float]
    notes: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "components": {k: round(v, 4) for k, v in self.components.items()},
            "notes": self.notes,
        }


def reward(svg_text: str, prompt: str | None = None) -> float:
    """Return a scalar reward in [0, 1]."""

    return score_svg(svg_text, prompt)["score"]


def score_svg(svg_text: str, prompt: str | None = None) -> dict[str, Any]:
    """Score one SVG and return total score plus interpretable components."""

    result = _score(svg_text or "", prompt or "")
    return result.as_dict()


def _score(raw_svg: str, prompt: str) -> RewardResult:
    notes: list[str] = []
    svg = _extract_svg(raw_svg)
    components: dict[str, float] = {}

    components["task_intent"] = _score_task_intent(raw_svg, notes)
    components["format"] = _score_format(raw_svg, svg, notes)
    # Merely mentioning an SVG snippet (for example while echoing the system
    # prompt) is not an attempt to perform the task.  Do not parse an embedded
    # ``<svg ...>...</svg>`` unless the response itself starts with <svg.
    if components["task_intent"] == 0.0:
        return _finalize(components, notes)

    try:
        root = ET.fromstring(svg)
        components["parse"] = 1.0
    except ET.ParseError as exc:
        notes.append(f"xml_parse_error: {exc}")
        components["parse"] = 0.0
        return _finalize(components, notes)

    elements = list(root.iter())
    components["svg_contract"] = _score_svg_contract(root, notes)
    components["safety"] = _score_safety(elements, notes)
    components["structure"] = _score_structure(elements, svg, notes)
    components["geometry"] = _score_geometry(elements, notes)
    components["palette"] = _score_palette(elements, notes)
    components["prompt_alignment"] = _score_prompt_alignment(prompt, elements, svg, notes)
    components["anti_degenerate"] = _score_anti_degenerate(svg, elements, notes)

    return _finalize(components, notes)


def _finalize(components: dict[str, float], notes: list[str]) -> RewardResult:
    weights = {
        "task_intent": 0.15,
        "format": 0.08,
        "parse": 0.14,
        "svg_contract": 0.10,
        "safety": 0.10,
        "structure": 0.12,
        "geometry": 0.11,
        "palette": 0.08,
        "prompt_alignment": 0.09,
        "anti_degenerate": 0.03,
    }
    # Missing components are failures, not reasons to renormalize the score.
    # This keeps malformed/non-SVG output below valid SVG output and prevents
    # early returns from receiving an artificially inflated reward.
    total = sum(
        weight * _clamp01(components.get(name, 0.0))
        for name, weight in weights.items()
    )
    score = total
    return RewardResult(score=_clamp01(score), components=components, notes=notes)


def _extract_svg(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:svg|xml)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    match = re.search(r"<svg\b[\s\S]*?</svg>", text, flags=re.IGNORECASE)
    return match.group(0).strip() if match else text


def _score_task_intent(raw: str, notes: list[str]) -> float:
    """Reward actually starting the requested artifact, not mentioning it."""

    stripped = raw.strip()
    if re.match(r"<svg\b", stripped, flags=re.IGNORECASE):
        return 1.0
    if re.search(r"<svg\b", stripped, flags=re.IGNORECASE):
        notes.append("svg_not_at_response_start")
    else:
        notes.append("no_svg_attempt")
    return 0.0


def _score_format(raw: str, svg: str, notes: list[str]) -> float:
    score = 0.0
    stripped = raw.strip()
    starts_with_svg = bool(re.match(r"<svg\b", stripped, flags=re.IGNORECASE))
    ends_with_svg = stripped.lower().endswith("</svg>")
    if starts_with_svg:
        score += 0.40
    else:
        notes.append("missing_opening_svg")
    if ends_with_svg:
        score += 0.30
    else:
        notes.append("missing_closing_svg")
    if starts_with_svg and ends_with_svg and stripped == svg:
        score += 0.20
    elif stripped != svg:
        notes.append("extra_text_outside_svg")
    if "```" not in raw:
        score += 0.10
    else:
        notes.append("markdown_fence_present")
    return score


def _score_svg_contract(root: ET.Element, notes: list[str]) -> float:
    score = 0.0
    if _tag(root.tag) == "svg":
        score += 0.25
    else:
        notes.append("root_is_not_svg")

    viewbox = root.attrib.get("viewBox") or root.attrib.get("viewbox")
    if viewbox and _normalize_space(viewbox) == "0 0 256 256":
        score += 0.35
    elif viewbox:
        score += 0.15
        notes.append(f"nonstandard_viewbox: {viewbox}")
    else:
        notes.append("missing_viewbox")

    xmlns = root.attrib.get("xmlns")
    if xmlns == "http://www.w3.org/2000/svg" or root.tag.startswith("{http://www.w3.org/2000/svg}"):
        score += 0.25
    else:
        notes.append("missing_or_nonstandard_xmlns")

    if not _has_external_reference(root):
        score += 0.15
    else:
        notes.append("external_reference_present")
    return score


def _score_safety(elements: list[ET.Element], notes: list[str]) -> float:
    bad_tags = sorted({_tag(el.tag) for el in elements if _tag(el.tag) in FORBIDDEN_TAGS})
    unknown_tags = sorted({_tag(el.tag) for el in elements if _tag(el.tag) not in ALLOWED_TAGS})
    event_attrs = [
        name
        for el in elements
        for name in el.attrib
        if name.lower().startswith("on") or "href" in name.lower()
    ]
    if bad_tags:
        notes.append(f"forbidden_tags: {bad_tags}")
    if unknown_tags:
        notes.append(f"unknown_tags: {unknown_tags}")
    if event_attrs:
        notes.append(f"unsafe_attrs: {sorted(set(event_attrs))}")
    tag_score = 1.0 - min(1.0, 0.25 * len(bad_tags) + 0.10 * len(unknown_tags))
    attr_score = 0.0 if event_attrs else 1.0
    return 0.7 * tag_score + 0.3 * attr_score


def _score_structure(elements: list[ET.Element], svg: str, notes: list[str]) -> float:
    graphics = [el for el in elements if _tag(el.tag) in GRAPHIC_TAGS]
    count = len(graphics)
    if count == 0:
        notes.append("no_graphic_elements")
        count_score = 0.0
    elif 3 <= count <= 80:
        count_score = 1.0
    elif count < 3:
        count_score = 0.45
        notes.append(f"too_few_graphic_elements: {count}")
    else:
        count_score = max(0.2, 1.0 - (count - 80) / 120)
        notes.append(f"too_many_graphic_elements: {count}")

    defs = sum(1 for el in elements if _tag(el.tag) == "defs")
    nested = sum(1 for el in elements if _tag(el.tag) == "g")
    size_score = 1.0 if 120 <= len(svg) <= 12000 else 0.45
    if len(svg) < 120:
        notes.append("svg_too_short")
    if len(svg) > 12000:
        notes.append("svg_too_long")
    group_score = 1.0 if defs <= 3 and nested <= 20 else 0.65
    return 0.60 * count_score + 0.25 * size_score + 0.15 * group_score


def _score_geometry(elements: list[ET.Element], notes: list[str]) -> float:
    values = _numeric_values(elements)
    if not values:
        notes.append("no_numeric_geometry")
        return 0.0

    finite = [v for v in values if math.isfinite(v)]
    if len(finite) != len(values):
        notes.append("nonfinite_numeric_value")
    if not finite:
        return 0.0

    in_soft_bounds = sum(-32 <= v <= 288 for v in finite) / len(finite)
    in_hard_bounds = sum(-4096 <= v <= 4096 for v in finite) / len(finite)
    spread = statistics.pstdev(finite) if len(finite) > 1 else 0.0
    spread_score = 1.0 if 5 <= spread <= 180 else 0.55
    if in_soft_bounds < 0.80:
        notes.append("many_coordinates_outside_256_canvas")
    if in_hard_bounds < 1.0:
        notes.append("extreme_coordinate_values")
    return 0.60 * in_soft_bounds + 0.25 * in_hard_bounds + 0.15 * spread_score


def _score_palette(elements: list[ET.Element], notes: list[str]) -> float:
    colors = _collect_colors(elements)
    if not colors:
        notes.append("no_explicit_colors")
        return 0.35
    if 2 <= len(colors) <= 8:
        palette_size = 1.0
    elif len(colors) == 1:
        palette_size = 0.65
        notes.append("single_color_palette")
    else:
        palette_size = max(0.25, 1.0 - (len(colors) - 8) / 16)
        notes.append(f"too_many_colors: {len(colors)}")

    invalid = [c for c in colors if not _looks_like_color(c)]
    valid_score = 1.0 - min(1.0, 0.25 * len(invalid))
    if invalid:
        notes.append(f"unusual_colors: {invalid[:5]}")
    return 0.65 * palette_size + 0.35 * valid_score


def _score_prompt_alignment(
    prompt: str, elements: list[ET.Element], svg: str, notes: list[str]
) -> float:
    if not prompt.strip():
        return 0.5

    lower_prompt = prompt.lower()
    lower_svg = svg.lower()
    tag_counts = {_tag(el.tag): 0 for el in elements}
    for el in elements:
        tag_counts[_tag(el.tag)] = tag_counts.get(_tag(el.tag), 0) + 1

    wanted_shapes = sorted({shape for word, shape in SHAPE_HINTS.items() if word in lower_prompt})
    shape_hits = 0
    for shape in wanted_shapes:
        if tag_counts.get(shape, 0) > 0:
            shape_hits += 1
    shape_score = shape_hits / len(wanted_shapes) if wanted_shapes else 0.5

    wanted_colors = sorted({word for word in COLOR_WORDS if re.search(rf"\b{re.escape(word)}\b", lower_prompt)})
    colors = _collect_colors(elements)
    color_hits = 0
    for word in wanted_colors:
        if _color_word_present(word, colors, lower_svg):
            color_hits += 1
    color_score = color_hits / len(wanted_colors) if wanted_colors else 0.5

    motif_words = _content_words(lower_prompt)
    motif_hits = sum(1 for word in motif_words if word in lower_svg)
    # Most motifs are visual concepts that will not appear literally in SVG. Keep
    # this term weak and saturating, so it cannot dominate validity.
    literal_score = min(1.0, motif_hits / 4) if motif_words else 0.5

    if wanted_shapes and shape_score < 0.5:
        notes.append(f"weak_shape_alignment: wanted={wanted_shapes}")
    if wanted_colors and color_score < 0.5:
        notes.append(f"weak_color_alignment: wanted={wanted_colors}")
    return 0.45 * shape_score + 0.45 * color_score + 0.10 * literal_score


def _score_anti_degenerate(svg: str, elements: list[ET.Element], notes: list[str]) -> float:
    lower = svg.lower()
    repeated_paths = len(re.findall(r"(m\s*[\d\-. ,]+){8,}", lower))
    repeated_chars = bool(re.search(r"([a-z0-9#<>/=\-])\1{30,}", lower))
    unique_tags = len({_tag(el.tag) for el in elements})
    score = 1.0
    if repeated_paths:
        score -= 0.25
        notes.append("repetitive_path_commands")
    if repeated_chars:
        score -= 0.35
        notes.append("repetitive_character_run")
    if unique_tags <= 2:
        score -= 0.20
        notes.append("low_tag_diversity")
    return _clamp01(score)


def _tag(name: str) -> str:
    return name.rsplit("}", 1)[-1] if "}" in name else name


def _normalize_space(value: str) -> str:
    return " ".join(value.replace(",", " ").split())


def _has_external_reference(root: ET.Element) -> bool:
    for el in root.iter():
        for value in el.attrib.values():
            lowered = value.strip().lower()
            if lowered.startswith(("http://", "https://", "javascript:", "data:")):
                return True
    return False


def _numeric_values(elements: list[ET.Element]) -> list[float]:
    values: list[float] = []
    for el in elements:
        for key, value in el.attrib.items():
            if key in {"id", "fill", "stroke", "class", "style", "transform"}:
                continue
            for match in re.finditer(r"[-+]?(?:\d*\.\d+|\d+)(?:e[-+]?\d+)?", value, re.IGNORECASE):
                try:
                    values.append(float(match.group(0)))
                except ValueError:
                    pass
    return values


def _collect_colors(elements: list[ET.Element]) -> list[str]:
    colors: list[str] = []
    for el in elements:
        for key, value in el.attrib.items():
            key_lower = key.lower()
            if key_lower in {"fill", "stroke", "stop-color"}:
                if value and not value.startswith("url(") and value.lower() != "none":
                    colors.append(value.strip().lower())
            elif key_lower == "style":
                for match in re.finditer(r"(?:fill|stroke|stop-color)\s*:\s*([^;]+)", value, re.IGNORECASE):
                    color = match.group(1).strip().lower()
                    if color and not color.startswith("url(") and color != "none":
                        colors.append(color)
    return sorted(set(colors))


def _looks_like_color(value: str) -> bool:
    if re.fullmatch(r"#[0-9a-fA-F]{3}(?:[0-9a-fA-F]{3})?", value):
        return True
    if re.fullmatch(r"rgba?\([^)]+\)", value):
        return True
    if value.lower() in COLOR_WORDS:
        return True
    return False


def _color_word_present(word: str, colors: list[str], lower_svg: str) -> bool:
    if word in lower_svg:
        return True
    expected = COLOR_WORDS.get(word)
    if expected and expected.lower() in colors:
        return True
    if expected:
        expected_rgb = _hex_to_rgb(expected)
        for color in colors:
            rgb = _hex_to_rgb(color)
            if rgb and expected_rgb and _rgb_distance(rgb, expected_rgb) <= 95:
                return True
    return False


def _hex_to_rgb(value: str) -> tuple[int, int, int] | None:
    match = re.fullmatch(r"#([0-9a-fA-F]{3}|[0-9a-fA-F]{6})", value.strip())
    if not match:
        return None
    raw = match.group(1)
    if len(raw) == 3:
        raw = "".join(ch * 2 for ch in raw)
    return int(raw[0:2], 16), int(raw[2:4], 16), int(raw[4:6], 16)


def _rgb_distance(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _content_words(lower_prompt: str) -> list[str]:
    stop = {
        "the",
        "and",
        "with",
        "inside",
        "center",
        "centered",
        "logo",
        "mark",
        "shape",
        "filled",
        "thin",
        "small",
        "large",
        "clean",
        "simple",
        "overall",
    }
    words = re.findall(r"[a-z][a-z\-]{3,}", lower_prompt)
    return sorted({w for w in words if w not in stop})[:20]


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


if __name__ == "__main__":
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("svg_file", nargs="?", help="SVG file to score. Reads stdin if omitted.")
    parser.add_argument("--prompt", default="")
    args = parser.parse_args()

    text = open(args.svg_file, encoding="utf-8").read() if args.svg_file else sys.stdin.read()
    print(json.dumps(score_svg(text, args.prompt), ensure_ascii=False, indent=2))
