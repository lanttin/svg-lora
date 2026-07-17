# SVG-LoRA

使用 LoRA 微调 Gemma 3 270M，根据详细文本描述生成 256×256 SVG 徽标。本仓库包含数据简化、可解释 reward、自评、checkpoint 选择脚本及训练完成的 LoRA adapter。

## 实验结果

在 17 条验证样本、贪心解码下：

| 模型 | Mean reward | 直接以 SVG 开始 | XML 可解析 | 达到 token 上限 |
|---|---:|---:|---:|---:|
| Base Gemma 3 270M | 0.0080 | 0/17 | 0/17 | 15/17 |
| LoRA | 0.2336 | 17/17 | 1/17 | 16/17 |

LoRA 明显学会了任务格式，但多数输出仍因重复生成而无法闭合。详细的数据处理、reward 权重、案例与原因分析见 [report.md](report.md)。

## 目录说明

```text
adapter/                         LoRA 权重
student_kit/reward.py            Reward 实现
student_kit/eval_self.py         基座与 LoRA 自评
util/simplify_svg_data.py        SVG 简化与渲染检查
train_config.yaml                最终训练配置
adapter_selection.json           Checkpoint 选择记录
results.json                     完整逐样本结果
gallery.html                     可视化对比页
```

## 快速复现

以下命令与 `RUN_COMMANDS.md` 一致，并应在仓库根目录分别执行。

1. 创建环境：

```bash
conda create -n svg-lora python=3.12 -y
conda activate svg-lora
python -m pip install -U pip
```

2. 安装依赖：

```bash
pip install -r requirements.txt
```

3. 检查 GPU：

```bash
python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA runtime:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
PY
```

4. 从 ModelScope 下载基座模型。请将占位符替换为实际模型仓库名：

```bash
modelscope download --model <gemma-3-270m repo> --local_dir ./gemma3-270m
```

5. 生成简化数据：

```bash
python util/simplify_svg_data.py
```

6. 训练 LoRA：

```bash
bash train_swift.sh
```

7. 选择验证 loss 最低的 checkpoint，并复制到 `adapter/`：

```bash
bash select_best_adapter.sh
```

8. 评测基座与 LoRA：

```bash
python student_kit/eval_self.py \
  --model ./gemma3-270m \
  --adapter adapter \
  --valid logo-detailed-prompt-simple/valid.jsonl \
  --output results.json \
  --max-new-tokens 1600 \
  --temperature 0
```

9. 生成可视化结果：

```bash
python student_kit/make_gallery.py --results results.json --output gallery.html
```

仅使用已提交 adapter 复现评测时，可跳过训练与 checkpoint 选择两步。更完整的环境与命令说明见 [RUN_COMMANDS.md](RUN_COMMANDS.md)。

## Reward

Reward 范围为 `[0,1]`，综合检查任务格式、XML 解析、SVG 合约、安全性、图元结构、坐标范围、调色板、提示词中的颜色/形状对齐及重复退化。它是训练代理指标，不等同于视觉审美评价；使用结果时应同时查看 XML parse rate 与截断率。

## 主要依赖

- PyTorch 2.8–2.9
- ModelScope ≥ 1.23
- ms-swift ≥ 4.0

本项目用于课程 Part B 实验。
