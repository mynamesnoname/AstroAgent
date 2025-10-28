# LLM-Spectrro-Agent

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
  - [ ] 最终的 main.py 中可能只包括 debug.ipynb 中的初始化阶段 + 对 workflow_orchestrator 函数的调用。