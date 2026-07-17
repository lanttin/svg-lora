# 基于 Gemma 3 270M 与 LoRA 的 SVG 徽标生成实验报告

## 1. 摘要

本项目使用 LoRA 对 Gemma 3 270M 进行监督微调，使模型根据详细英文描述生成单个、完整且安全的 SVG 徽标。针对 270M 小模型容易复述提示词、输出不闭合以及陷入重复的问题，我首先把原始 SVG 数据转换为更浅、更短的纯图元表示，并设计了一个由格式、XML 有效性、安全性、结构、几何、配色、提示词对齐和抗退化等部分组成的可解释 reward。

在 17 条验证样本、贪心解码的最终自评中，基座模型的平均 reward 为 **0.0080**，LoRA 模型为 **0.2336**，绝对提升 **+0.2256**。LoRA 将“回答直接以 `<svg>` 开始”的比例从 **0/17 提升到 17/17**，说明模型确实学会了任务格式；但只有 **1/17** 的回答能作为完整 XML 解析，且 **16/17** 达到 1600 token 上限。因而本实验的真实结论不是“已经会画高质量徽标”，而是“已从完全偏离任务提升到稳定尝试 SVG，但仍严重受重复生成和闭合失败限制”。

## 2. 任务与仓库结构

任务目标是比较同一个 Gemma 3 270M 基座在微调前后的相对变化，主要提交物如下：

- `adapter/`：step 200 的 LoRA 权重与配置；
- `student_kit/reward.py` 与根目录 `reward.py`：reward 实现及提交入口；
- `train_config.yaml`：最终训练超参数；
- `results.json`：固定设置下的逐样本基座/LoRA 自评结果；
- `report.md`：实验报告。


## 3. 数据预处理

### 3.1 为什么需要简化

原始数据包含渐变、滤镜、裁剪、mask、`<use>`、动画和很大的背景矩形。它们对人工 SVG 很有用，却增加了小模型要学习的 XML 层级、引用关系和序列长度。早期方案只按总 token 数过滤原始数据（219 条中保留 201 条），仍没有消除这些结构性难点。最终方案改为“先语义尽量保留地简化，再按复杂度过滤”。

### 3.2 最终处理流程

`util/simplify_svg_data.py` 对每个 assistant SVG 执行以下确定性处理：

1. 用渐变的第一个 stop color 替换 `url(#...)` paint server；
2. 展开可解析的本地 `<use>` 引用；
3. 仅保留 `svg/g/path/circle/ellipse/rect/polygon/line`；
4. 删除 defs、filter、clipPath、mask、动画、text、事件属性和外部引用；
5. 将异常大的全画布背景矩形归一化到 `0 0 256 256`；
6. 小数属性保留两位，并统一 `viewBox="0 0 256 256"`；
7. 重新做 XML 解析，并通过 ImageMagick 实际渲染检查；
8. 训练集中剔除 SVG 超过 3200 字符或图形元素超过 40 个的样本。

同时，system prompt 被统一成与最终生成约束一致的版本：只输出一个 SVG、只用允许的纯色图元、最多 40 个图形元素，并必须闭合。

### 3.3 处理统计

| 划分 | 原始条数 | 最终条数 | 过滤条数 | 最终 SVG 平均字符数 | 图元中位数 / 最大值 | 渲染失败 |
|---|---:|---:|---:|---:|---:|---:|
| Train | 219 | 205 | 14 | 1339.60 | 14 / 40 | 0 |
| Valid | 17 | 17 | 0 | 1518.47 | 15 / 40 | 0 |

训练集累计替换 318 个渐变、展开 112 个 `<use>`，并归一化 107 个背景；验证集替换 22 个渐变并归一化 7 个背景。所有 236 个转换后的 SVG 均通过 XML 与渲染检查。完整统计见 `logo-detailed-prompt-simple/simplify_report.json`。

这一处理降低了目标序列的结构熵，但有明确代价：渐变被压成单色、裁剪和滤镜被移除，视觉细节会损失；过滤长样本也让训练分布更偏向简单徽标。这是为了优先让 270M 模型学会“完整输出”所做的工程取舍。

## 4. Reward 设计

### 4.1 设计原则

reward 是训练代理指标，不是真实视觉评价。对极小模型而言，首先应奖励“确实开始完成 SVG 任务、语法有效、安全、边界合理”，再给予较轻的语义分。若过早把复杂视觉语义置于最高权重，分数会被不可可靠测量的部分主导。

总分范围为 `[0,1]`，缺失的组件按 0 分处理而不重新归一化：

| 组件 | 权重 | 检查内容与理由 |
|---|---:|---|
| task_intent | 0.15 | 回答必须直接以 `<svg>` 开始，防止模型复述 system/user prompt 后仅提及 SVG |
| format | 0.08 | 开头、结尾、单一文档、无 Markdown fence，保证可直接使用 |
| parse | 0.14 | `ElementTree` XML 解析成功；这是可用 SVG 的基础 |
| svg_contract | 0.10 | 根节点、xmlns、256×256 viewBox、无外部引用 |
| safety | 0.10 | 禁止 script、image、foreignObject、事件属性及不支持标签 |
| structure | 0.12 | 3–80 个图元、合理文本长度和浅层分组，兼顾非空与不过度复杂 |
| geometry | 0.11 | 数值有限，多数坐标位于画布附近，无极端值且具有一定分布 |
| palette | 0.08 | 颜色合法，优先 2–8 色的小而连贯的调色板 |
| prompt_alignment | 0.09 | 根据提示词中的颜色词、形状词和少量字面 motif 做弱匹配 |
| anti_degenerate | 0.03 | 检查重复 path/字符与过低标签多样性 |

reward 采用“门控”逻辑：若输出不是从 `<svg>` 开始，立即结束；若 XML 解析失败，也不再计算依赖 DOM 的后续组件。这能避免一段复述文本因为包含 `<svg>...</svg>` 示例而得到高分，也避免解析失败样本凭局部特征获得虚高分。

### 4.2 语义对齐的实现与边界

形状对齐将 circle、square、star、leaf 等提示词映射到预期 SVG 标签；颜色对齐既检查颜色名/hex，也允许 RGB 距离在 95 内的近似色；motif 字面命中只占语义子分的 10%。这种设计可解释、无需外部模型，但无法判断某条 `<path>` 是否真的像树叶、手掌或瓶子，也无法衡量构图、遮挡、对称性和审美。


## 5. 训练与复现流程

### 5.1 最终配置

| 项目 | 设置 |
|---|---|
| 基座模型 | `./gemma3-270m` |
| 框架 | ms-swift SFT |
| 精度 | bfloat16 |
| 训练 / 验证集 | simple 205 / 17 |
| Epoch | 8 |
| Batch size / 梯度累积 | 1 / 8（有效 batch 8） |
| 学习率 / warmup | 1e-4 / 0.05 |
| LoRA rank / alpha / dropout | 8 / 16 / 0.05 |
| Target modules | all-linear（导出配置中为 q/k/v/o、gate/up/down projection） |
| 最大训练长度 | 2048 |
| eval/save 间隔 | 25 steps |
| 随机种子 | 42 |

### 5.2 实验迭代

本项目依次尝试了三类数据方案：原始数据；仅过滤超过 2048 token 的数据；最终的纯图元简化数据。保留的历史结果显示，在“仅长度清洗”的一次评测中，`max_new_tokens=1900` 时基座/LoRA reward 为 **0.008/0.190**，两者 17/17 均触达上限。最终简化数据实验变为 **0.008/0.2336**，说明目标简化带来了一些额外收益，但重复与停止问题仍未解决。由于这些迭代同时改变了数据和生成长度，不能把差值严格解释为单一变量的消融；它们仅作为开发轨迹，最终结论以当前 `results.json` 为准。

### 5.3 checkpoint 选择

验证 loss 从 step 25 的 0.9531 持续下降，在 step 200 达到最低 **0.7519**，step 208 小幅回升到 0.7524。因此按“现存 checkpoint 中 eval_loss 最低”选择 step 200，而不是机械使用最后一步。这一拐点虽很小，但符合早停思路。

### 5.4 从头复现

以下过程与 `RUN_COMMANDS.md` 保持一致，所有命令均在仓库根目录执行。

**步骤 1：创建并进入 Python 3.12 环境。**

```bash
conda create -n svg-lora python=3.12 -y
conda activate svg-lora
python -m pip install -U pip
```

**步骤 2：安装项目依赖。**

```bash
pip install -r requirements.txt
```

**步骤 3：确认 PyTorch、CUDA 与 GPU 状态。**

```bash
python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA runtime:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
PY
```

**步骤 4：从 ModelScope 下载 Gemma 3 270M。** 需要将占位符替换为实际的 ModelScope 模型仓库名，下载后的目录必须是 `./gemma3-270m`，以匹配训练配置。

```bash
modelscope download --model <gemma-3-270m repo> --local_dir ./gemma3-270m
```

**步骤 5：重新生成简化后的训练集和验证集。**

```bash
python util/simplify_svg_data.py
```

该命令会创建或刷新 `logo-detailed-prompt-simple/train.jsonl`、`valid.jsonl` 和 `simplify_report.json`。

**步骤 6：按照 `train_config.yaml` 训练 LoRA。**

```bash
bash train_swift.sh
```

**步骤 7：根据验证 loss 选择适配器。**

```bash
bash select_best_adapter.sh
```

脚本从最新训练目录的现存 checkpoint 中选择 `eval_loss` 最低者，并将权重复制到 `adapter/`。

**步骤 8：在完整验证集上评测基座与 LoRA。**

```bash
python student_kit/eval_self.py \
  --model ./gemma3-270m \
  --adapter adapter \
  --valid logo-detailed-prompt-simple/valid.jsonl \
  --output results.json \
  --max-new-tokens 1600 \
  --temperature 0
```

**步骤 9：生成可视化对比页。**

```bash
python student_kit/make_gallery.py --results results.json --output gallery.html
```

打开 `gallery.html` 即可并排查看 Target、Base、Adapter 及对应 reward。正式完整评测前，也可以先运行以下烟测命令检查模型和 adapter 是否能正常加载：

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

自评使用 temperature 0、top_p 1.0 与固定验证集，因此解码是确定性的；硬件、PyTorch/ms-swift 小版本差异仍可能导致轻微变化。更详细的命令见 `RUN_COMMANDS.md`。

## 6. 最终结果

### 6.1 定量结果

| 指标 | Base | LoRA | 变化 |
|---|---:|---:|---:|
| 平均 reward | 0.0080 | 0.2336 | **+0.2256** |
| 直接以 `<svg>` 开始 | 0/17 | 17/17 | +17 |
| XML 解析成功 | 0/17 | 1/17 | +1 |
| 找到闭合 SVG 片段 | 17/17* | 2/17 | -15 |
| 触达生成 token 上限 | 15/17 | 16/17 | +1 |
| 平均输出 token | 1460.1 | 1576.9 | +116.8 |

\* 基座的“闭合片段”是它复述 system prompt 中的字面示例 `<svg>...</svg>`，并非完成了任务，所以 reward 的 task_intent 正确地把这些样本判为 0。该指标必须与“是否从 SVG 开始”联合阅读。

LoRA 的逐样本 reward 分布为：1 条 0.8905、15 条 0.19、1 条 0.23。均值被唯一高分样本明显拉高；中位数只有 **0.19**。若只看均值，会高估改进的稳定性。

### 6.2 定性案例

**案例 A：儿童绘画徽标（index 0）**

- Base：复述完整 system prompt 与用户描述，随后只出现字面 `<svg>...</svg>` 占位，reward 0.008。
- LoRA：正确以标准根节点开头，使用 cream 背景、rect/circle/path 等合法图元，第一个 SVG 片段可解析，reward 0.8905。
- 问题：大量重复相同圆形，闭合后又继续生成重复 path，视觉内容与“画笔、彩带、闪光”只弱相关。高分主要来自合法前缀、标准画布和安全结构，不代表高视觉质量。

**案例 B：救援之家徽标（index 1）**

- Base：重复提示词而不真正作答，reward 0.008。
- LoRA：能抽取提示中的 `#E8ECEF`、`#1F4E79`、`#F2994A`，并从 circle/path 开始构图，说明有局部颜色条件能力；但迅速退化为 `L128 128` 循环，耗尽 1600 token，XML 未闭合，reward 0.19。

**案例 C：验证集 index 16**

- LoRA 未触达 token 上限且带 `</svg>`，但属性引号损坏（`fill="none stroke=...`），仍不能解析，reward 0.23。
- 这表明“生成了结束标签”不是充分条件；XML 级约束必须覆盖整个解码过程。

完整输出很长，不适合在报告中全文复制，可直接打开 `gallery.html` 或查看 `results.json` 逐项核验。

## 7. 为什么会得到这样的结果

1. **SFT 学到了强而局部的格式模式。** 205 个训练答案都从标准 `<svg>` 根节点开始，因此 LoRA 很容易把首 token 行为从“复述提示词”改为“立即输出 SVG”。这解释了 task_intent 从 0% 到 100%，也是 reward 上升的主要来源。
2. **学会开头比学会长程闭合容易。** 一份 SVG 通常需要上千 token，闭合依赖模型持续维护引号、标签和 path 状态。270M 参数量与 2048 训练长度下，局部 token 模式可以拟合，长程结构一致性仍困难。
3. **确定性解码会固化循环。** temperature 0 便于公平复现，但模型一旦进入高概率的 `circle`、`L128 128` 或 path 片段，就没有随机性帮助其跳出循环。于是 16/17 LoRA 输出耗尽上限。
4. **复杂详细提示与小模型容量不匹配。** 每条提示同时要求对象、颜色、层次、风格和象征意义。模型偶尔能复制 hex 色值和基础形状，却难以把全部条件组织为一致构图。
5. **数据简化降低了语法难度，但未直接教停止。** 目标长度变短且标签更少，使 reward 从历史 0.190 提升到 0.2336；然而训练目标仍包含较长 path，且没有专门的闭合/重复负样本或结构约束，因此循环问题依旧。
6. **验证 loss 与生成质量不完全一致。** loss 衡量 teacher-forcing 下下一个 token 的平均概率；推理时一个错误会改变后续上下文并累积。step 200 的 loss 较好，只说明对参考序列的局部预测改善，并不保证自由生成时能闭合。


## 8. 局限与未来改进


### 8.1 数据与训练改进

- 进一步限制目标长度与 path 复杂度，把复杂 path 拆成少量可学习的基础图元；显式加入短小、正确闭合的模板样本。
- 加入课程学习：先训练极短的 3–8 图元图标，再逐步增加元素数量和提示复杂度。

### 8.2 解码改进

- 在首次生成合法 `</svg>` 后立即停止，并禁止继续产生尾随文本。
- 尝试更小的 `max_new_tokens` 与结构感知截断/修复，但修复结果应单独评估，不能与模型原始能力混淆。

## 9. 结论

本项目证明 LoRA 能让 Gemma 3 270M 从“几乎只复述任务”转为“稳定以标准 SVG 根节点开始”，代理 reward 获得 +0.2256 的明显相对提升；但长序列闭合、抗重复与语义构图仍是主要瓶颈。当前结果最有价值的洞察是：对极小模型，格式学习可以很快成功，而 teacher-forcing loss、代理 reward 和真实可渲染/视觉质量之间存在明显鸿沟。未来改进重点是尝试提高svg正确闭合率。
