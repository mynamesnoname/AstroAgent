这是一张从天文学光谱中提取出的连续谱图像。

## 连续谱趋势分析

请用纯定性语言判断：

1. 蓝端、中段、红端三段的相对通量水平排序（高 / 中 / 低）。
2. 蓝端 → 中段的整体趋势（上升 / 下降）。
3. 中段 → 红端的整体趋势（上升 / 下降）。

请严格按以下 JSON 格式输出结果：

{
  "blue_end": "high" | "medium" | "low",
  "blue_to_mid_trend": "increasing" | "decreasing",
  "mid_section": "high" | "medium" | "low",
  "mid_to_red_trend": "increasing" | "decreasing",
  "red_end": "high" | "medium" | "low"
}

禁止输出任何其他内容。
