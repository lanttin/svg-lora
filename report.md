# SVG-LoRA Report

## Reward design

My reward prioritizes valid, safe, bounded SVG output before semantic fidelity. This is appropriate because Gemma 3 270M is small; producing a complete parseable SVG is already a meaningful capability.

Components:

- Format: output should be exactly one `<svg>...</svg>` document, without prose or markdown fences.
- Parse: SVG must parse as XML.
- SVG contract: root should be `<svg>`, include `xmlns`, and use `viewBox="0 0 256 256"`.
- Safety: disallow scripts, images, external references, event handlers, and unsupported tags.
- Structure: reward a reasonable number of vector primitives.
- Geometry: reward finite numeric values mostly inside or near the 256 by 256 canvas.
- Palette: reward small cohesive palettes with valid color values.
- Prompt alignment: lightly reward requested shapes and colors appearing in the SVG representation.
- Anti-degeneration: penalize repetitive character runs, repetitive paths, and low tag diversity.

## Training setup

- Base model: `./gemma3-270m`
- Framework: ms-swift
- Method: LoRA SFT
- Train data: `logo-detailed-prompt/train.jsonl`
- Validation data: `logo-detailed-prompt/valid.jsonl`
- Epochs: 8
- Learning rate: 1e-4
- LoRA rank: 8
- LoRA alpha: 16
- Max length: 2048
- Batch size: 1
- Gradient accumulation: 8

## Results

Fill after running `python student_kit/eval_self.py`.

| Model | Mean reward |
|---|---:|
| Base Gemma 3 270M | TODO |
| LoRA adapter | TODO |
| Delta | TODO |

## Qualitative examples

Add several prompts from `results.json`, with base output and adapter output. Focus on validity, SVG structure, and whether the major requested shapes/colors appear.

## Analysis

TODO: Explain whether the adapter improved over the base model. Mention common failure modes such as malformed SVG, missing closing tags, overlong output, weak prompt fidelity, or repetitive shapes.

## Reward limitations

The reward is not a true visual judge. It can verify syntax, structure, safety, geometry, and rough color/shape evidence, but it cannot reliably tell whether a path actually looks like a leaf, column, nut, or musical note. Therefore a higher reward should be interpreted as stronger SVG discipline, not necessarily better visual design.
