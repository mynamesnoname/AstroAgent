中文 README 文件见 [README in Chinese](./README_Chinese.md).

# LLM-Spectro-Agent

An LLM-powered agent for human-like analysis of one-dimensional astronomical spectra.

## Overview

This project use large language models (LLMs) to perform human-like astrophysical inference on 1D spectra, specifically:
- **Source classification** (Only support star, galaxy, QSO)
- **Redshift estimation**

The system mimics the cognitive workflow of a human astronomer:
1. **Visual interpretation** of the spectrum plot (axes, units, features)
2. **Rule-based analysis** using astrophysical knowledge (e.g., Lyα, C IV, Mg II lines)
3. **Multi-agent debate** between an auditor and refinement assistant to improve robustness
4. **Synthesis** of a final report with confidence assessment

The pipeline is currently configured to use the following Qwen models via API:
- **Text reasoning**: `qwen3-max-2025-09-23`
- **Visual understanding**: `qwen-vl-max-2025-08-13`

> ⚠️ Note: Other LLMs have not been tested and may require adaptation.

---

## Dependencies & Installation

### 1. Python Dependencies
Install the required Python packages:
```bash
pip install -r requirements.txt
```

### 2. System Dependencies
This project relies on **Tesseract OCR** for text detection in spectrum plots. Install it based on your OS:

- **Ubuntu/Debian**:
  ```bash
  sudo apt-get install tesseract-ocr
  ```

- **macOS** (with Homebrew):
  ```bash
  brew install tesseract
  ```

- **Windows**:  
  Download and install from [UB Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki)

> 📌 Make sure `tesseract` is in your system PATH. Verify with:
> ```bash
> tesseract --version
> ```

### 3. Environment Setup
Copy the example configuration and fill in your settings:
```bash
cp .env_example .env
```

Edit `.env` to specify:
- `API_KEY`: Your API key for LLM models
- `INPUT_DIR`, `OUTPUT_DIR`: Input and output directories
- `IMAGE_NAME`: Name of the spectrum image (without `.png` extension)
- and other parameters

---

## Quick Start

### 1. Run the Analysis
Execute the main script:
```bash
python main.py
```
Results will be saved to the output directory specified in `.env`.

### 2. Try the Notebook (Optional)
For interactive exploration and debugging, see `debug2.ipynb`. After setting up your environment variables, you can run this notebook to step through the analysis pipeline.

---

## Test set
A basic test set is offered in ./data/test_set

---

## Output Files

For an input image named `{your_image_name}.png`, the program generates the following outputs in the configured output directory:

- `{your_image_name}_cropped.png`  
  Cleaned spectrum image with titles, axes, and borders removed.

- `{your_image_name}_reconstructed.png`  
  Reconstructed spectrum after OpenCV-based preprocessing.

- `{your_image_name}_features.png`  
  Visualization of detected peaks and troughs.

- `{your_image_name}_rule_analysis.md`  
  Intermediate rule-based analysis from the spectral analyst agent.

- `{your_image_name}_summary.md`  
  Final synthesized report including source type, redshift estimate, and confidence assessment.

---

## Architecture Highlights

- **Multi-Agent Debate**: Uses LangGraph to orchestrate a structured debate between an auditor and refinement assistant, enhancing result reliability.
- **Multi-Scale Feature Detection**: Detects spectral features across multiple Gaussian smoothing scales and merges them robustly.
- **Hybrid Vision + Language**: Combines computer vision (OpenCV, OCR) with multimodal LLMs for end-to-end spectrum understanding.
- **MCP Integration**: Built on the Model Context Protocol (MCP) for standardized tool interaction.

---

## Requirements

```txt
# Core scientific computing
numpy>=1.21.0
scipy>=1.7.0
pandas>=1.3.0

# Computer vision & OCR
opencv-python>=4.5.0
pytesseract>=0.3.8

# Plotting
matplotlib>=3.4.0

# Language model & agent framework
langchain-openai>=0.1.0
langgraph>=0.1.0
langchain-core>=0.1.0

# MCP (Model Context Protocol) integration
langchain-mcp-adapters>=0.1.0
mcp>=1.19.0

# Utilities
python-dotenv>=0.19.0
openai>=1.0.0
pydantic>=2.0.0
```
---

## License

This project is for research and educational purposes. 

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