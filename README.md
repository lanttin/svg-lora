# SVG-LoRA

使用 LoRA 微调 Gemma 3 270M，根据详细英文描述生成 256×256 SVG 徽标。仓库包含 SVG 数据简化、可解释 reward、基座/LoRA 自评、checkpoint 选择脚本和训练完成的 adapter。

## 实验结果

最终自评使用原始数据集验证集 `logo-detailed-prompt/valid.jsonl`，共 17 条样本；采样参数为 `temperature=1.0`、`top_p=1.0`、`seed=42`，最大生成长度为 1600 token。

| 模型 | Mean reward | 直接以 SVG 开始 | XML 严格可解析 | 浏览器可解析/展示 | 达到 token 上限 |
|---|---:|---:|---:|---:|---:|
| Base Gemma 3 270M | 0.0030 | 0/17 | 0/17 | 0/17 | 5/17 |
| LoRA | 0.4588 | 17/17 | 8/17 | 16/17 | 9/17 |

LoRA 已稳定学会直接生成 SVG，并有 8 条输出获得 0.83–0.97 的 reward。浏览器对部分非严格 XML 存在容错，因此浏览器可展示率高于 XML 严格解析率；该栏是人工结合画廊检查得到的补充指标，不参与 reward。

可以直接打开 [可视化画廊](gallery.html)，并排查看 Target、Base、Adapter 和逐样本 reward。完整数值与原始输出保存在 [results.json](results.json)，详细方法和分析见 [report.md](report.md)。

## 数据与评测口径

- 训练：`logo-detailed-prompt-simple/train.jsonl`，205 条简化样本。
- 训练期验证与 checkpoint 选择：`logo-detailed-prompt-simple/valid.jsonl`，17 条简化样本。
- 最终推理自评：`logo-detailed-prompt/valid.jsonl`，17 条原始样本。

最终自评读取原始验证集中的 system、user 和 assistant 消息：system/user 用于模型推理，原始 assistant SVG 只作为 Target 展示，不输入模型。训练期使用简化验证集，是为了让验证 loss 与简化训练目标保持一致；最终自评使用原始验证集，则用于观察模型在原始任务指令和参考图上的表现。

## 目录说明

```text
adapter/                         LoRA 权重
student_kit/reward.py            Reward 实现
student_kit/eval_self.py         基座与 LoRA 自评
student_kit/make_gallery.py      画廊生成脚本
util/simplify_svg_data.py        SVG 简化与渲染检查
train_config.yaml                训练配置
adapter_selection.json           Checkpoint 选择记录
results.json                     完整逐样本结果
gallery.html                     可视化画廊
report.md                        实验报告
```

## 快速复现

以下命令在仓库根目录执行。

1. 创建环境并安装依赖：

```bash
conda create -n svg-lora python=3.12 -y
conda activate svg-lora
python -m pip install -U pip
pip install -r requirements.txt
```

2. 下载基座模型：

```bash
modelscope download --model <gemma-3-270m repo> --local_dir ./gemma3-270m
```

3. 生成简化数据、训练并选择 checkpoint：

```bash
python util/simplify_svg_data.py
bash train_swift.sh
bash select_best_adapter.sh
```

4. 使用原始验证集进行最终自评：

```bash
python student_kit/eval_self.py \
  --model ./gemma3-270m \
  --adapter adapter \
  --valid logo-detailed-prompt/valid.jsonl \
  --output results.json \
  --max-new-tokens 1600 \
  --temperature 1.0 \
  --top-p 1.0 \
  --seed 42
```

5. 生成画廊：

```bash
python student_kit/make_gallery.py --results results.json --output gallery.html
```

更完整的环境和运行命令见 [RUN_COMMANDS.md](RUN_COMMANDS.md)。

## Reward

Reward 范围为 `[0,1]`，权重与 `student_kit/reward.py` 一致：

| 组件 | 权重 |
|---|---:|
| task_intent | 0.05 |
| format | 0.03 |
| parse | 0.10 |
| svg_contract | 0.07 |
| safety | 0.08 |
| structure | 0.15 |
| geometry | 0.14 |
| palette | 0.11 |
| prompt_alignment | 0.20 |
| anti_degenerate | 0.07 |

输出必须直接以 `<svg>` 开始，才会进入 XML 检查；严格 XML 解析失败后，不再计算依赖 DOM 的质量项。浏览器容错展示不属于 reward。Reward 是可解释的训练代理指标，不等同于视觉审美评价，因此应同时查看严格 XML 解析率、浏览器展示率、截断率和 [画廊](gallery.html)。

## 主要依赖

- PyTorch 2.8–2.9
- ModelScope ≥ 1.23
- ms-swift ≥ 4.0

本项目用于课程 Part B 实验。
