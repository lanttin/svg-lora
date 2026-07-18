# SVG-LoRA run commands

## 1. 创建环境

```bash
conda create -n svg-lora python=3.12 -y
conda activate svg-lora
python -m pip install -U pip
```

安装依赖

```bash
pip install -r requirements.txt
```

验证 GPU 是否可用：

```bash
python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA runtime:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
PY
```

## 2. 下载基座模型

```bash
modelscope download --model <gemma-3-270m repo> --local_dir ./gemma3-270m
```

## 3. 快速检查 reward

如果需要重新生成清洗后的数据目录，运行：

```bash
python util/simplify_svg_data.py
```

这会创建或刷新：

```text
logo-detailed-prompt-simple/
  train.jsonl
  valid.jsonl
  simplify_report.json
```

## 4. 训练 LoRA：ms-swift

```bash
bash train_swift.sh
```

训练完成后，从最新训练目录的现存 checkpoint 中选择 `eval_loss` 最低的权重，
并复制到 `adapter/`：

```bash
bash select_best_adapter.sh
```

## 5. 自评：基座 vs 微调

```bash
python student_kit/eval_self.py \
  --model ./gemma3-270m \
  --adapter adapter \
  --valid logo-detailed-prompt/valid.jsonl \
  --output results.json \
  --max-new-tokens 1600 \
  --temperature 1.0 \
  --seed 42
```

生成一个可以用浏览器打开的可视化对比页：

```bash
python student_kit/make_gallery.py --results results.json --output gallery.html
```

然后打开 `gallery.html`，可以并排看 Target、Base、Adapter 的 SVG 和 reward。

只跑前 3 条做烟测：

```bash
python student_kit/eval_self.py \
  --model ./gemma3-270m \
  --adapter adapter \
  --valid logo-detailed-prompt-simple/valid.jsonl \
  --output results_smoke.json \
  --limit 3 \
  --max-new-tokens 1600 \
  --temperature 1.0 \
  --seed 42
```
