"""Create a local HTML gallery from eval_self.py results."""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results.json")
    parser.add_argument("--output", default="gallery.html")
    args = parser.parse_args()

    results = json.loads(Path(args.results).read_text(encoding="utf-8"))
    base_items = results.get("runs", {}).get("base", {}).get("items", [])
    adapter_items = results.get("runs", {}).get("adapter", {}).get("items", [])
    adapter_by_index = {item["index"]: item for item in adapter_items}

    rows = []
    for base in base_items:
        idx = base["index"]
        adapter = adapter_by_index.get(idx)
        rows.append(
            render_row(
                idx=idx,
                prompt=base.get("prompt", ""),
                target=base.get("target", ""),
                base_svg=base.get("prediction", ""),
                base_score=base.get("reward", {}).get("score", 0),
                adapter_svg=adapter.get("prediction", "") if adapter else "",
                adapter_score=adapter.get("reward", {}).get("score", 0) if adapter else None,
            )
        )

    document = HTML_TEMPLATE.replace("<!--ROWS-->", "\n".join(rows))
    Path(args.output).write_text(document, encoding="utf-8")
    print(f"Wrote {args.output}")


def render_row(
    *,
    idx: int,
    prompt: str,
    target: str,
    base_svg: str,
    base_score: float,
    adapter_svg: str,
    adapter_score: float | None,
) -> str:
    adapter_title = "Adapter"
    adapter_score_text = "n/a" if adapter_score is None else f"{adapter_score:.4f}"
    return f"""
<section class="case">
  <h2>#{idx}</h2>
  <p class="prompt">{html.escape(prompt)}</p>
  <div class="grid">
    {panel("Target", target, None)}
    {panel("Base", base_svg, base_score)}
    {panel(adapter_title, adapter_svg, adapter_score_text)}
  </div>
</section>
"""


def panel(title: str, svg_text: str, score) -> str:
    clean = extract_svg(svg_text)
    score_html = "" if score is None else f"<span>reward: {score}</span>"
    if not clean:
        body = '<div class="empty">No SVG</div>'
    else:
        body = f'<div class="svgbox">{clean}</div>'
    return f"""
<article class="panel">
  <header><strong>{html.escape(title)}</strong>{score_html}</header>
  {body}
  <details>
    <summary>SVG source</summary>
    <pre>{html.escape(svg_text[:4000])}</pre>
  </details>
</article>
"""


def extract_svg(text: str) -> str:
    match = re.search(r"<svg\b[\s\S]*?</svg>", text or "", flags=re.IGNORECASE)
    return match.group(0) if match else ""


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SVG-LoRA Gallery</title>
<style>
body {
  margin: 0;
  font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: #f6f7f9;
  color: #1b1f24;
}
main {
  max-width: 1280px;
  margin: 0 auto;
  padding: 24px;
}
h1 {
  margin: 0 0 18px;
  font-size: 28px;
}
.case {
  padding: 18px 0 26px;
  border-top: 1px solid #d8dde5;
}
.case h2 {
  margin: 0 0 8px;
  font-size: 18px;
}
.prompt {
  max-width: 1100px;
  color: #3d4652;
  line-height: 1.45;
}
.grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 14px;
}
.panel {
  background: white;
  border: 1px solid #d8dde5;
  border-radius: 8px;
  overflow: hidden;
}
.panel header {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  padding: 10px 12px;
  border-bottom: 1px solid #e8ebf0;
  font-size: 14px;
}
.svgbox {
  display: grid;
  place-items: center;
  min-height: 300px;
  padding: 18px;
  background:
    linear-gradient(45deg, #eef1f5 25%, transparent 25%),
    linear-gradient(-45deg, #eef1f5 25%, transparent 25%),
    linear-gradient(45deg, transparent 75%, #eef1f5 75%),
    linear-gradient(-45deg, transparent 75%, #eef1f5 75%);
  background-size: 20px 20px;
  background-position: 0 0, 0 10px, 10px -10px, -10px 0;
}
.svgbox svg {
  width: min(100%, 256px);
  height: auto;
  max-height: 280px;
}
.empty {
  min-height: 300px;
  display: grid;
  place-items: center;
  color: #8a94a3;
}
details {
  border-top: 1px solid #e8ebf0;
  padding: 8px 12px;
}
summary {
  cursor: pointer;
  font-size: 13px;
}
pre {
  white-space: pre-wrap;
  word-break: break-word;
  font-size: 12px;
}
@media (max-width: 920px) {
  .grid {
    grid-template-columns: 1fr;
  }
}
</style>
</head>
<body>
<main>
  <h1>SVG-LoRA Gallery</h1>
  <!--ROWS-->
</main>
</body>
</html>
"""


if __name__ == "__main__":
    main()

