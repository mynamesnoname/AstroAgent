# LLM-Spectro-Agent

一个由大语言模型（LLM）驱动的智能体，用于对一维天文光谱进行**类人式分析与推断**。

> 📄 相关论文正在准备中。

---

## 项目概述（Overview）

本项目利用大语言模型（LLMs）对一维天文光谱（1D spectra）进行类似人类天文学家的物理推断，当前支持的核心任务包括：

- **天体类型分类**（目前仅支持：恒星 / 星系 / 类星体 QSO）
- **红移估计**

系统设计目标是**模拟人类天文学家的认知流程**，主要包括以下步骤：

1. **光谱图的视觉理解**  
   自动解析坐标轴、单位、整体形态与显著特征
2. **基于规则的物理分析**  
   结合天体物理知识（如 Lyα、C IV、Mg II 等谱线）
3. **多智能体辩论机制**  
   审计代理（auditor）与修正代理（refinement assistant）之间进行推理对抗，以增强鲁棒性
4. **综合总结输出**  
   生成最终分析报告，并给出置信度评估

---

## 使用的模型（Models）

当前流程通过 API 使用以下 **Qwen 系列模型**：

- **文本推理模型**：`qwen3-max-2025-09-23`
- **视觉理解模型**：`qwen-vl-max-2025-08-13`

> ⚠️ 注意：其他大语言模型尚未测试，如需替换可能需要适配。

---

## 依赖与安装（Dependencies & Installation）

### 1. OCR 引擎

项目支持两种 OCR（光学字符识别）后端：

- **PaddleOCR（默认）**
- **Tesseract OCR**

默认使用 PaddleOCR，因为其在坐标轴刻度等图像文本识别方面通常更准确，但安装流程相对复杂。

在 `src/utils` 中提供了两个 OCR 封装函数：

```python
_detect_axis_ticks_paddle(state)
```

使用 PaddleOCR。

```python
_detect_axis_ticks_tesseract(state)
```

使用 Tesseract OCR。

你可以在 `.env` 文件中配置所使用的 OCR 引擎。

---

### 1.1 安装 PaddleOCR

PaddleOCR 依赖 PaddlePaddle，需要先安装 PaddlePaddle。

#### （1）安装 PaddlePaddle

**仅 CPU 版本：**

```bash
pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
```

如需 GPU 支持或系统相关说明，请参考 PaddlePaddle [官方安装页面](https://www.paddlepaddle.org.cn/)。

#### （2）安装 PaddleOCR

```bash
pip install "paddleocr[all]"
```

#### （3）LangChain 兼容性修复（重要）

当前 PaddleOCR 使用了旧版 LangChain 的导入方式，而本项目依赖新版：

* `langchain-core`
* `langchain-text-splitter`

因此需要在安装完成后**手动修补 PaddleOCR 源码**。

打开以下文件（路径请根据你的 Conda 环境调整）：

```bash
nano ~/Apps/anaconda3/envs/your_env_name/lib/python3.12/site-packages/paddlex/inference/pipelines/components/retriever/base.py
```

将：

```python
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
```

替换为：

```python
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
```

💡 提示：可通过以下命令定位环境路径：

```bash
which python
# 或
conda info --envs
```

---

### 1.2 安装 Tesseract OCR

根据操作系统选择：

* **Ubuntu / Debian**

  ```bash
  sudo apt-get install tesseract-ocr
  ```

* **macOS（Homebrew）**

  ```bash
  brew install tesseract
  ```

* **Windows**
  请从以下地址下载安装：
  [https://github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki)

确认安装成功：

```bash
tesseract --version
```

---

## 2. Python 依赖

安装 Python 依赖包：

```bash
pip install -r requirements.txt
```

---

### 3. MCP 配置
将 `configs/mcp_config.json` 中的:
```json
"args": ["/data2/wbc/llm-spectro-agent/src/AstroAgent/mcp_tools/spectro_server.py"],  
```
改为对应路径 (即项目的 `/src/AstroAgent/mcp_tools/spectro_server.py`)

---

## 4. 环境变量配置

复制示例配置文件：

```bash
cp .env_example .env
```

并编辑 `.env`，主要参数包括：

* `API_KEY`：LLM API 密钥
* `INPUT_DIR`：输入图像目录
* `OUTPUT_DIR`：输出结果目录
* `IMAGE_NAME`：光谱图像名（不含 `.png`）
* 以及其他控制参数

---

## 快速开始（Quick Start）

详见 [Quick Start](Quickstart.md)

### 运行分析

```bash
python main.py
```

分析结果将保存至 `.env` 中指定的输出目录。

---

## 测试数据集（Test Set）

在以下路径提供了基础测试集：

```text
./test_set/CSST/input
./test_set/DESI/input
```
使用时请将 .env 中的 `INPUT_DIR` 设置为该路径。

---

## 输出文件说明（Output Files）

对于输入图像 `{your_image_name}.png`，程序会生成主要产品：

* `{your_image_name}_spec_extract.png`
  基于 OpenCV 的重建光谱

* `{your_image_name}_features.png`
  检测到的谱峰与谱谷可视化结果

* `{your_image_name}_continuum.png`
  拟合后的连续谱

* `{your_image_name}_rule_analysis.md`
  基于规则的中间分析报告

* `{your_image_name}_summary.md`
  最终综合报告（天体类型、红移、置信度）

* in_brief.csv
  输出结果摘要的 CSV 格式表格

以及副产品

* `{your_image_name}_cropped.png`
  去除标题、坐标轴、边框后的纯光谱图

* `{your_image_name}_ocr_res_img.png`
  OCR 结果可视化图（仅Paddle版本输出）

* `{your_image_name}_res.json`
  OCR 结果文本（仅Paddle版本输出）

* `{your_image_name}_spectrum.png`
  提取到的光谱与 SNR 图

* `{your_image_name}_visual_interpretation.txt`
  视觉分析的中间产物

---

## 系统架构亮点（Architecture Highlights）

* **多智能体辩论机制**
  使用 LangGraph 协调审计与修正代理，提升结果可靠性

* **多尺度谱线特征检测**
  在多个高斯平滑尺度下检测谱峰并稳健合并

* **视觉 + 语言混合架构**
  结合 OpenCV、OCR 与多模态 LLM，实现端到端光谱理解

* **MCP 协议集成**
  基于 Model Context Protocol（MCP）进行标准化工具调用

---

## 许可证（License）

本项目仅用于**科研与教学目的**。

---

## 更新日志

- 2025.12.24: remove environment variables `WEIGHT_ORIGINAL`