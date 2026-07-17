# SVG-LoRA run commands

下面命令从零开始，假设当前目录是本仓库根目录。

## 1. 创建环境



```bash
conda create -n svg-lora python=3.12 -y
conda activate svg-lora
python -m pip install -U pip
```

安装依赖
//--extra-index-url https://download.pytorch.org/whl/cu128
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

作业要求优先用 ModelScope。把 `<gemma-3-270m repo>` 替换成老师指定或 ModelScope 页面上的模型 id。

```bash
modelscope download --model <gemma-3-270m repo> --local_dir ./gemma3-270m
```

如果 `modelscope download` 在你的版本里不可用，可以用 Python API：

```bash
python -c "from modelscope import snapshot_download; snapshot_download('<gemma-3-270m repo>', local_dir='./gemma3-270m')"
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

当前规则是：渐变转纯色、展开 `<use>`、移除复杂效果、规范化背景矩形，
并真实渲染检查全部结果；训练集中超过 3200 字符或 40 个图形元素的复杂样本会被删除。

```bash
python - <<'PY'
import json
from student_kit.reward import score_svg
row = json.loads(open("logo-detailed-prompt-simple/train.jsonl", encoding="utf-8").readline())
prompt = next(m["content"] for m in row["messages"] if m["role"] == "user")
svg = next(m["content"] for m in row["messages"] if m["role"] == "assistant")
print(json.dumps(score_svg(svg, prompt), ensure_ascii=False, indent=2))
PY
```

## 4. 训练 LoRA：ms-swift

```bash
CUDA_VISIBLE_DEVICES=0 swift sft \
  --model ./gemma3-270m \
  --tuner_type lora \
  --dataset logo-detailed-prompt-simple/train.jsonl \
  --val_dataset logo-detailed-prompt-simple/valid.jsonl \
  --torch_dtype bfloat16 \
  --num_train_epochs 8 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --learning_rate 1e-4 \
  --lora_rank 8 \
  --lora_alpha 16 \
  --target_modules all-linear \
  --gradient_accumulation_steps 8 \
  --eval_steps 25 \
  --save_steps 25 \
  --save_total_limit 20 \
  --logging_steps 5 \
  --max_length 2048 \
  --output_dir output/swift-svg-lora \
  --warmup_ratio 0.05
```

也可以直接执行同内容脚本：

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
  --valid logo-detailed-prompt-simple/valid.jsonl \
  --output results.json \
  --max-new-tokens 1600 \
  --temperature 0
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
  --temperature 0
```

## 6. 最终提交目录

```text
adapter/
  adapter_config.json
  adapter_model.safetensors
reward.py
train_config.yaml
results.json
report.md
```
