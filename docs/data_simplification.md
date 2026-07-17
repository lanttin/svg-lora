# SVG 训练数据简化记录

## 背景与问题

第一轮 LoRA 训练后，模型已经能够稳定地以正确的 SVG 根标签开始输出：

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256">
```

但验证集中的输出全部达到生成长度上限，且没有生成闭合的 `</svg>`。主要退化模式包括：

- 重复生成 `<stop>` 渐变节点；
- 重复相同的 path 命令，例如 `L128 128`；
- 重复相同的 `<line>` 或 `<circle>`；
- 忘记关闭 `<linearGradient>`、`<radialGradient>` 或 `<defs>`；
- 因耗尽 `max_new_tokens` 而留下不完整 XML。

原始训练集只有 219 条数据，却要求 270M 参数模型同时学习详细语义、SVG
几何规划、复杂 XML 嵌套、渐变和滤镜结构。对于该规模的模型，目标结构过于复杂。

## 原始数据观察

原始训练集的 219 条 SVG 均可通过 XML 解析，但包含大量复杂结构：

- 201 条包含 `<defs>`；
- 193 个 `<linearGradient>`；
- 125 个 `<radialGradient>`；
- 731 个 `<stop>`；
- 112 个 `<use>`；
- 还有 clipPath、filter、mask、animation 和 SVG filter primitive；
- 107 条包含 `x="-9999"`, `width="19998"` 一类超大背景矩形。

这些结构本身可以合法渲染，但会增加小模型需要正确记忆和关闭的 XML 层级，
并在训练数据中形成很强的高频模板。

## 简化目标

数据简化优先保证以下目标：

1. 每条目标仍然是合法、可渲染的 SVG；
2. 保留 Logo 的主要形状、颜色和构图；
3. 只保留浅层、容易闭合的 SVG 标签；
4. 消除容易触发重复退化的 gradient/stop/defs 结构；
5. 统一 system prompt、训练答案和 reward 的 SVG 约束；
6. 控制答案长度和图形元素数量。

## 实现方式

简化脚本位于：

```text
util/simplify_svg_data.py
```

脚本使用 `xml.etree.ElementTree` 解析和修改 SVG，而不是使用正则表达式直接处理
XML 层级。处理步骤如下。

### 渐变转纯色

收集每个 linear/radial gradient 的第一个 `stop-color`，然后将：

```xml
fill="url(#sunGrad)"
```

替换为：

```xml
fill="#F2994A"
```

stroke 引用使用相同规则。若渐变不存在或没有有效 stop-color，则使用基于 SHA-256
稳定选择的 fallback 颜色，保证不同进程重复清洗时结果一致。

### 展开 `<use>`

对于本地 `href="#id"` 引用，复制被引用的 shape/group，并合并 x、y 和 transform。
这样可以删除 `<defs>`，同时尽量保留 `<use>` 所表示的可见图形。

### 删除复杂结构

最终只允许：

```text
svg, g, path, circle, ellipse, rect, polygon, line
```

删除 defs、gradient、stop、filter、clipPath、mask、style、animation、text、image
等结构，同时移除 filter、clip-path、mask、href 和事件属性。

### 规范化背景和数值

将覆盖整个画布的异常超大背景矩形：

```xml
<rect x="-9999" y="-9999" width="19998" height="19998" .../>
```

改为：

```xml
<rect x="0" y="0" width="256" height="256" .../>
```

小数属性最多保留两位，以减少无意义的 token。

### 复杂样本过滤

训练样本满足任一条件时删除：

- 简化后的 assistant SVG 超过 3200 字符；
- 图形元素超过 40 个。

验证集不根据复杂度删除，以保持固定的 17 条验证样本。

## System prompt 同步

简化后的 system prompt 明确要求：

- 只输出一个完整 SVG；
- 只使用白名单图形标签；
- 只使用纯色 fill/stroke；
- 禁止 defs、gradient、filter、mask、clipPath 和 use；
- 最多使用 40 个图形元素；
- 必须以 `</svg>` 结束。

这避免了旧 system prompt 鼓励使用 gradients/filters，而训练答案又删除这些结构的矛盾。

## 最终数据统计

训练集：

```text
原始样本：219
保留样本：205
过滤样本：14
```

训练答案字符数：

| 指标 | 字符数 |
|---|---:|
| 最小 | 323 |
| 中位数 | 1255 |
| 平均 | 1339.6 |
| P90 | 2146.4 |
| P95 | 2398.2 |
| 最大 | 3040 |

训练答案图形元素数：

| 指标 | 数量 |
|---|---:|
| 最小 | 3 |
| 中位数 | 14 |
| 平均 | 15.91 |
| P90 | 30 |
| P95 | 33 |
| 最大 | 40 |

验证集保留全部 17 条，答案字符数中位数为 1286、最大为 3274。

## 正确性验证

所有 236 条源数据在简化后均进行了两层验证：

1. 使用 ElementTree 再次解析 XML；
2. 使用 ImageMagick 将 SVG 实际渲染为 PNG。

结果为：

```text
XML 解析成功：236/236
实际渲染成功：236/236
渲染失败：0
```

最终保留的 205 条训练目标和 17 条验证目标也确认：

- 不包含 `url(#...)`；
- 不包含 `-9999` 背景坐标；
- 不包含白名单外标签；
- 全部以 `</svg>` 结束；
- reward XML parse 分量全部为 1。

目标 SVG 的平均程序化 reward 为：

```text
训练集：0.9759
验证集：0.9769
```

## 长度配置

训练继续使用：

```yaml
max_length: 2048
```

推理推荐：

```text
max_new_tokens: 1600
```

1600 对简化后的最长 SVG 留有余量，又能比 1900 更早暴露重复退化。云端训练前应使用
实际 Gemma tokenizer 再检查一次：

```bash
python util/token_stats.py \
  --data logo-detailed-prompt-simple/train.jsonl \
  --model ./gemma3-270m
```

## 局限

- 渐变被替换为单色，视觉层次会降低；
- 删除 clipPath/filter 后，个别图形可能出现轻微视觉变化；
- 展开 use 可能增加部分 SVG 的元素数量；
- 可渲染只能证明 SVG 技术上有效，不能证明与 prompt 完全一致；
- 当前数据量仍然较少，简化结构不能完全替代更多高质量训练数据。

完整机器可读统计见：

```text
logo-detailed-prompt-simple/simplify_report.json
```
