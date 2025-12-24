中文 README 文件见 [README in Chinese](./README_Chinese.md).

 **A related paper is in preparation.** 

# LLM-Spectro-Agent

An LLM-powered agent for human-like analysis of one-dimensional astronomical spectra.

## Overview

This project use large language models (LLMs) to perform human-like astrophysical inference on 1D spectra, specifically:
- **Source classification** [Only support galaxy (LRG and ELG, output as galaxy), QSO]
- **Redshift estimation** for QSOs

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
### 1. OCR Engine

This project supports two OCR (Optical Character Recognition) backends: PaddleOCR and Tesseract OCR.
By default, PaddleOCR is used because it generally offers higher accuracy—especially for chart axis labels—but requires a more involved installation process.

In src/utils, we provide two OCR wrapper functions:
```python
_detect_axis_ticks_paddle(state)
```
Uses PaddleOCR.
```python
_detect_axis_ticks_tesseract(state)
```
Uses Tesseract OCR.

You can select your preferred OCR engine by setting the appropriate option in your .env file.

#### 1.1 Installing PaddleOCR

PaddleOCR depends on PaddlePaddle, which must be installed first.

##### 1. Install PaddlePaddle
For CPU-only support, run:

```bash
pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
```

For GPU support or detailed instructions (including system-specific guidance), refer to the [official PaddlePaddle installation page](https://www.paddlepaddle.org.cn/).

##### 2. Install PaddleOCR
```bash
pip install "paddleocr[all]"
```
##### 3. Compatibility Fix for LangChain (you can do it after installing the Python dependencies)
The current version of PaddleOCR uses legacy imports from older versions of LangChain (langchain.docstore.document, etc.), while this project relies on the newer
* `langchain-core`
* `langchain-text-splitter`

To resolve this conflict, you’ll need to manually patch the PaddleOCR source code after installing the Python dependencies (see Section 2).

Open the following file in your editor (adjust the path to match your Conda environment):
```bash
nano ~/Apps/anaconda3/envs/your_env_name/lib/python3.12/site-packages/paddlex/inference/pipelines/components/retriever/base.py
```
Replace these lines:
```python
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
```
with:
```python
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
```
💡 Tip: You can locate your environment path using 
```bash
which python
```
or 
```bash
conda info --envs.
```
#### 1.2 Installing Tesseract OCR

Install it based on your OS:

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

### 2. Python Dependencies
Install the required Python packages:
```bash
pip install -r requirements.txt
```

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
For interactive exploration and debugging, see `debug.ipynb`. After setting up your environment variables, you can run this notebook to step through the analysis pipeline.

---

## Test set
A basic test set is offered in ./data/test_set

---

## Output Files Description

For an input image `{your_image_name}.png`, the program will generate main products:

* `{your_image_name}_spec_extract.png`
  Reconstructed spectrum based on OpenCV.

* `{your_image_name}_features.png`
  Visualization of detected spectral peaks and troughs.

* `{your_image_name}_continuum.png`
  Fitted continuum spectrum.

* `{your_image_name}_rule_analysis.md`
  Intermediate rule-based analysis report.

* `{your_image_name}_summary.md`
  Final comprehensive report (object type, redshift, confidence level).

And by-products:

* `{your_image_name}_cropped.png`
  Clean spectrum image with titles, axes, and borders removed.

* `{your_image_name}_ocr_res_img.png`
  OCR result visualization image (output only by Paddle version).

* `{your_image_name}_res.json`
  OCR result text (output only by Paddle version).

* `{your_image_name}_spectrum.png`
  Extracted spectrum and SNR plot.

* `{your_image_name}_visual_interpretation.txt`
  Intermediate product of visual analysis.

---

## Architecture Highlights

- **Multi-Agent Debate**: Uses LangGraph to orchestrate a structured debate between an auditor and refinement assistant, enhancing result reliability.
- **Multi-Scale Feature Detection**: Detects spectral features across multiple Gaussian smoothing scales and merges them robustly.
- **Hybrid Vision + Language**: Combines computer vision (OpenCV, OCR) with multimodal LLMs for end-to-end spectrum understanding.
- **MCP Integration**: Built on the Model Context Protocol (MCP) for standardized tool interaction.

---
## License

This project is for research and educational purposes. 

---
Update 
- 2025.12.24: remove environment variables `WEIGHT_ORIGINAL`