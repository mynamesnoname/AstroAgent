# LLM-Spectro-Agent

一个基于大语言模型（LLM）的智能体，仿照人类思维流程对一维天文光谱进行分析。

## 概述

本项目利用大语言模型（LLM）对一维天文光谱执行类人化的天体物理推断，目前支持：
- **天体分类**（仅支持恒星、星系、类星体/QSO）
- **红移估计**

系统模拟人类天文学家的认知流程：
1. **视觉解读**：识别光谱图中的坐标轴、单位和特征；
2. **规则分析**：结合天体物理知识（如 Lyα、C IV、Mg II 谱线）进行推理；
3. **多智能体辩论**：由审查员与优化助手进行多轮辩论，提升结果鲁棒性；
4. **综合报告**：生成包含置信度评估的最终分析结论。

当前默认配置通过 API 调用以下通义千问（Qwen）模型：
- **文本推理**：`qwen3-max-2025-09-23`
- **视觉理解**：`qwen-vl-max-2025-08-13`

> ⚠️ 注意：其他大模型尚未测试，如需使用可能需要适配。

---

## 依赖与安装

### 1. Python 依赖
安装所需 Python 包：
```bash
pip install -r requirements.txt
```

### 2. 系统依赖
本项目依赖 **Tesseract OCR** 从光谱图像中识别文本（如坐标轴刻度）。请根据操作系统安装：

- **Ubuntu/Debian**：
  ```bash
  sudo apt-get install tesseract-ocr
  ```

- **macOS**（使用 Homebrew）：
  ```bash
  brew install tesseract
  ```

- **Windows**：  
  从 [UB Mannheim Tesseract 官方页面](https://github.com/UB-Mannheim/tesseract/wiki) 下载并安装。

> 📌 请确保 `tesseract` 已加入系统 `PATH`。可通过以下命令验证：
> ```bash
> tesseract --version
> ```

### 3. 环境配置
复制示例配置文件并填写你的设置：
```bash
cp .env_example .env
```

编辑 `.env` 文件，配置以下关键参数：
- `DASHSCOPE_API_KEY`：通义千问 DashScope 平台的 API 密钥；
- `INPUT_DIR`, `OUTPUT_DIR`：输入与输出目录路径；
- `IMAGE_NAME`：光谱图像文件名（不含 `.png` 后缀）；
- 其他可选参数

---

## 快速开始

### 1. 运行分析
执行主程序：
```bash
python main.py
```
结果将保存到 `.env` 中指定的输出目录。

### 2. 使用 Notebook 调试（可选）
项目包含交互式调试笔记本 `debug2.ipynb`。配置好环境变量后，可逐步运行分析流程，便于开发与排查问题。

---

## 输出文件说明

对于输入图像 `{your_image_name}.png`，程序将在输出目录中生成以下文件：

- `{your_image_name}_cropped.png`  
  裁剪后的光谱图像（已移除标题、坐标轴和边框）。

- `{your_image_name}_reconstructed.png`  
  经 OpenCV 预处理后重建的光谱图像。

- `{your_image_name}_features.png`  
  可视化检测到的峰值（发射线）与谷值（吸收线）。

- `{your_image_name}_rule_analysis.md`  
  光谱规则分析智能体生成的中间分析报告。

- `{your_image_name}_summary.md`  
  最终综合报告，包含天体类型、红移估计值及置信度评估。

---

## 架构亮点

- **多智能体辩论机制**：基于 LangGraph 编排审查员与优化助手的结构化辩论，显著提升结果可靠性；
- **多尺度特征检测**：在多个高斯平滑尺度下检测光谱特征，并进行鲁棒融合；
- **视觉+语言混合推理**：结合计算机视觉（OpenCV、OCR）与多模态大模型，实现端到端光谱理解；
- **MCP 协议集成**：基于 Model Context Protocol（MCP）标准，实现工具调用的规范化与可扩展性。

---

## 依赖清单（requirements.txt）

```txt
# 核心科学计算
numpy>=1.21.0
scipy>=1.7.0
pandas>=1.3.0

# 计算机视觉与 OCR
opencv-python>=4.5.0
pytesseract>=0.3.8

# 绘图
matplotlib>=3.4.0

# 大模型与智能体框架
langchain-openai>=0.1.0
langgraph>=0.1.0
langchain-core>=0.1.0

# MCP（Model Context Protocol）集成
langchain-mcp-adapters>=0.1.0
mcp>=1.19.0

# 工具库
python-dotenv>=0.19.0
openai>=1.0.0
pydantic>=2.0.0
```

---

## 许可声明

本项目仅用于**科研与教育目的**。

---

## 10.29 更新
- 新增 main.py
  - 使用方式：
    - 配置 .env 文件。示例为 .env_example
    - 而后运行
      ```bash
      python main.py
      ```

- 使用 langgraph 对 流程进行了改写。旧的 astro_agents 程序在 src/_astro_agents_old.py 中
- 新增 src/workflow_orchestrator.py，用来管理 agent 的运行流程

## 10.24 更新
- 使用环境变量作为参数的输入方式
- 环境变量的配置在 .env 文件中。.env 文件的配置示例在 .env_example 文件里。
- 接受这些输入参数的位置：
  - debug.ipynb 的初始化阶段，接受
    - input_dir = os.getenv('INPUT_DIR')
    - output_dir = os.getenv('OUTPUT_DIR')
    - SINGLE_RUN = os.getenv('SINGLE_RUN').lower()=='true'
    - image_name = os.getenv('IMAGE_NAME')
    - IMAGE_NAME_HEADER、START、END这三个参数是批量处理所使用的参数，暂未实装
  - src/mcp_manager._init_llm() 接受两种 LLM 的参数。llm_type='LLM' 或 'VIS_LLM'.
    - api_key = self._get_env_or_raise(f"{llm_type}_API_KEY")
    - base_url = self._get_env_or_raise(f"{llm_type}_BASE_URL").rstrip()
    - model = os.getenv(f"{llm_type}_MODEL", default_model)
    - temp_str = os.getenv(f"{llm_type}_TEMPERATURE", "0.1")
    - temperature = float(temp_str) if temp_str else 0.1
    - max_tokens_str = os.getenv(f"{llm_type}_MAX_TOKENS")
  - src/astro_agent.SpectralVisualInterpreter.run() 接受
    - SIGMA_LIST
    - TOL_PIXELS
    - WEIGHT_ORIGINAL
    - PROM_THRESHOLD_PEAKS
    - PROM_THRESHOLD_TROUGHS
    - 以及 
      - p_ = os.getenv('PLOT_PEAKS_NUMBER')
      - t_ = os.getenv('PLOT_TROUGHS_NUMBER')

- 下一步计划：
  - [x] 将 debug.ipynb 中目前的运行流程封装到 src.workflow_orchestrator
  - [x] 最终的 main.py 中可能只包括 debug.ipynb 中的初始化阶段 + 对 workflow_orchestrator 函数的调用。