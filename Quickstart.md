# Quick Start

This project aims to build a LLM agent for human-like analysis of one-dimensional astronomical spectra, inspired by the process of Visual Inspection. The agent is designed to perform tasks such as source classification (QSO or Galaxy) and redshift estimation for QSOs.

The agent is built using the LangChain framework, which provides a high-level interface for interacting with LLMs. It uses the OpenAI API for LLM inference. We test this agent using the Qwen3 model, other LLMs may require some adaptation.

## Usage

After cloning the repository, and installing the required Python packages (see `requirements.txt` and `Readme.md`), you need to set up your environment variables in `.env`. 

By running 
```bash
cp .env_example .env
```
you can copy the example configuration and fill in your settings.

Below are some important environment variables:

- `MCP_CONFIG`: The path to the MCP configuration file. A default configuration is provided in `configs/mcp_config.json`

- `PROMPTS_CONFIG_PATH`: The path to the prompt configuration file. A default configuration is provided in `configs/prompt_config_EN.json` and `configs/prompt_config_CN.json`

- `PROMPTS_ROOT`: The root directory for prompts contents. Our prompts are stored in the `prompt_content` directory in format of `.md` files. Just set it to `/path/to/your/project/prompt_content`

- `RUN_MODE`: This project provide two modes: `s` for single-run mode and `b` for batch mode.

- `INPUT_DIR`: The directory where the input images are stored.
- `OUTPUT_DIR`: The directory where the output files will be saved.
- `IMAGE_NAME`: The name of the input image, without the `.png` extension.
    - e.g. If `INPUT_DIR`=`/path/to/your/project/input`, `OUTPUT_DIR`=`/path/to/your/project/output`, `IMAGE_NAME`=`your_image_name`, then the input image will be `/path/to/your/project/input/your_image_name.png`. And the output files will be saved to `/path/to/your/project/output/`.

- `BATCH_HEADER`, `BATCH_START`, `BATCH_END`: These variables are used to specify the range of images to process in batch mode. 
    - e.g. If `BATCH_HEADER`=`example_`, `BATCH_START`=`1`, `BATCH_END`=`10`, then the input images will be `/path/to/your/project/input/example_1.png`, `/path/to/your/project/input/example_2.png`, ..., `/path/to/your/project/input/example_10.png`.
    - If `BATCH_START`=`01`, `BATCH_END`=`10`, then the input images will be `/path/to/your/project/input/example_01.png`, `/path/to/your/project/input/example_02.png`, ..., `/path/to/your/project/input/example_10.png`.
    - You can leave `BATCH_HEADER` empty if your input images are named `1.png`, `2.png`, ..., `10.png`.

- `DATASET`: This variable is used to specify the dataset. It is related with the prompts. The prompts in `prompt_content` directory is modified according to the dataset instrument and spectrum characteristics. Now we support `DESI` and `CSST`.
- `OCR`: This variable is used to specify the OCR engine. We support `Paddle OCR` (recommended) and `Tesseract OCR`.

After setting up your environment variables, you can run the agent by running the following command:
```bash
python scripts/main.py
```
and the results will be saved to the output directory.

## File Structure
The project is structured as follows:
```txt
AstroAgent
├── .env_example
├── .gitignore
├── pyproject.toml
├── Quickstart.md
├── README_Chinese.md
├── README.md
├── requirements.txt
├── configs
│   ├── mcp_config.json
│   ├── prompt_config_CN.json
│   └── prompt_config_EN.json
├── notebooks # some notebooks for testing
│   ├── llm_test.ipynb
│   ├── mcp_test.ipynb
│   ├── multi_agents_test.ipynb
│   ├── prompts_test.ipynb
│   ├── prompts.ipynb
│   └── runtime_test.ipynb
├── prompt_content # prompts in markdown format
│   ├── CN
│   └── EN 
├── scripts
│   └── main.py
├── src
│   └── AstroAgent
│       ├── agents
│       │   ├── common
│       │   │   ├── base_agent.py
│       │   │   ├── result_writer.py
│       │   │   ├── state.py
│       │   │   └── utils.py
│       │   └── multi_agents
│       │       ├── AnalysisAuditor.py
│       │       ├── RefinementAssistant.py
│       │       ├── RuleAnalyst.py
│       │       ├── SynthesisHost.py
│       │       └── VisualInterpreter.py
│       ├── core
│       │   ├── config
│       │   │   ├── all_config.py
│       │   │   ├── batch_config.py
│       │   │   ├── io_config.py
│       │   │   ├── mcp_config.py
│       │   │   ├── model_config.py
│       │   │   ├── params_config.py
│       │   │   └── prompt_config.py
│       │   ├── runtime
│       │   │   └── runtime_container.py
│       │   └── llm.py
│       ├── manager
│       │   ├── mcp
│       │   │   └── mcp_manager.py
│       │   └── runtime
│       │       ├── message_manager.py
│       │       ├── prompt_manager.py
│       │       └── state_manager.py
│       ├── mcp_tools
│       │   ├── spectro_server.py
│       │   ├── tool_protocol.py
│       │   └── tools.py
│       └── workflow_orchestrator.py
└── test_set
```

## Program Structure

The entry point of the program is `scripts/main.py`. In this file, we use `AllConfig` from `src/AstroAgent/core/configs/all_config.py` to load all the environment variables in `.env`. 

After that, `scripts/main.py` initailizes the `RuntimeContainer` from `src/AstroAgent/core/runtime/runtime_container.py`. It is responsible for loading the model, creating the MCP client, and creating the prompt manager.

The program then initializes the `WorkflowOrchestrator` from `src/AstroAgent/workflow_orchestrator.py`. This class is responsible for orchestrating the workflow of the program. Since we use `langgraph` to build the agent, it require a `langgraph state` class to transfer the state between agents. The initial state is created by the `PromptManager` from `src/AstroAgent/manager/prompt.py`.

Then the `WorkflowOrchestrator` calls the `run` method, which starts the workflow. The workflow steps is defined in its `_create_workflow()` function.

All the corresponding agents are defined in `src/AstroAgent/agents/`.
