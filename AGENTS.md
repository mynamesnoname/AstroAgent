# AGENTS.md — FORMA

Guidance for AI coding agents working in this repository. Assume the reader knows nothing about the project.

## Project Overview

**FORMA** is an LLM-powered multi-agent system for human-like analysis of one-dimensional astronomical spectra (DESI FITS files). It performs:

- **Source classification** — galaxy (LRG / ELG / BGS), QSO, star
- **Redshift estimation** for QSOs

It mimics a human astronomer's visual-inspection workflow: deterministic feature extraction (CWT wavelet feature detection, continuum fitting) → redshift hypothesis generation via **Redrock** (DESI's official redshift fitter, run as an external subprocess) → per-hypothesis LLM agent evaluation in parallel → adversarial LLM audits → final report with calibrated confidence.

The pipeline is built on **LangChain / LangGraph** (Python ≥ 3.12) and calls an OpenAI-compatible chat API (developed and tested against `deepseek-v4-pro`; Qwen is also supported via vendor detection). A related academic paper is in preparation (see `.paper/`, git-ignored).

## Repository Layout

```
├── scripts/
│   ├── main.py                # CLI entry point (async, single & batch modes)
│   └── webui.py               # Gradio WebUI (3 tabs: Config / Run / Results); wraps CLI pipeline via subprocess
├── src/FORMA/                 # Installable package (src-layout, pip install -e .)
│   ├── workflow_orchestrator.py   # LangGraph StateGraph wiring all agents; retry/cancel logic
│   ├── core/
│   │   ├── config/            # Pydantic config models loaded from .env (all/io/batch/model/params)
│   │   ├── llm.py             # ChatOpenAI factory; vendor detection (deepseek/qwen), thinking-mode
│   │   │                      #   extra_body, DeepSeek max_tokens payload patch
│   │   └── runtime/runtime_container.py  # Lazy model-client cache shared by agents
│   └── agents/
│       ├── common/            # BaseAgent (retry/timeout handling), SpectroState (LangGraph MessagesState),
│       │                      #   SpectroStateFactory, ResultWriter, message_utils
│       └── multi_agents/
│           ├── VisualInterpreter.py    # Stage 1: FITS loading, continuum fit, CWT feature detection,
│           │                             #   Redrock run (subprocess), redshift scoring
│           ├── HypothesisAnalyst.py    # Stage 2 (parallel per-hypothesis harness) + Stage 4 (synthesis)
│           ├── AnalysisAuditor.py      # Stage 3 FeatureAuditor + Stage 5 ResultAuditor (adversarial review)
│           ├── ReportWriter.py         # Stage 6: final_report.md + in_brief JSON
│           ├── SelfEvolve.py           # Optional ground-truth comparison & failure root-cause analysis
│           ├── harness/           # LangGraph agent harness: single_hypothesis.py, hypothesis_synthesis.py,
│           │                        #   result_auditor.py, tools.py (LLM tools: read_spectrum_region, grep_kb, …),
│           │                        #   continuation.py (truncated-response continuation)
│           │   ├── kb/            # Astrophysics knowledge base in Markdown (lines, classification, ionization)
│           │   └── skills/        # Per-agent prompt/skill Markdown files
│           └── utils/           # VI.py, HA.py, SE.py, cwt_feature_finder.py, plot.py, usage.py
├── example/                   # Ready-to-run inputs: basic/ (5 DESI VI-Q4 spectra), counter_fact_examples/
├── test_set/DESI/             # FITS files grouped by class (BGS/ELG/LRG/QSO) — git-ignored data
├── Dockerfile, docker-compose.yml, docker-compose.dev.yml, docker/entrypoint.sh
├── pyproject.toml             # Package metadata + dependencies (setuptools, src-layout)
├── requirements.txt           # Pinned deps (used by Dockerfile; some entries commented out, e.g. dashscope, MCP)
├── .env_example               # Annotated template for all configuration (bilingual EN/CN)
├── package.json               # Only for Playwright (used by .repo_info HTML tooling); NOT part of the pipeline
└── .repo_info/                # Generated HTML module docs (index.html) — reference only
```

## Workflow Architecture

The LangGraph workflow is defined in `WorkflowOrchestrator._create_workflow()` (`src/FORMA/workflow_orchestrator.py`):

```
START → visual_interpreter → (has features?) ──no──→ no_features (placeholder report) → END
                              │yes
                              ▼
              hypothesis_analyst_search  (parallel harness, HARNESS_CONCURRENCY)
                    → feature_auditor → hypothesis_analyst_synthesize
                    → analysis_auditor → report_writer → END
```

- State flows through `SpectroState` (`src/FORMA/agents/common/state.py`), a LangGraph `MessagesState` subclass; all fields are `Optional` and agents read/write state keys directly.
- The orchestrator retries the whole workflow on connection/timeout errors (`MAX_TRIES` / `RETRY_DELAY`), restarting from a deep copy of the initial state to avoid dirty list-append fields.
- `SelfEvolve` is **not** in the graph as a fixed node — the batch failure analysis is triggered from `scripts/main.py` when `SELF_EVOLVE=true` and enough failures accumulate (`FAILURE_BATCH_SIZE`).

## Configuration

All runtime configuration comes from **environment variables** via `.env` (copy `.env_example` → `.env`; it is fully annotated). Loaded by `AllConfig.from_env()` → `IOConfig`, `BatchConfig`, `ModelConfig`, `ParamsConfig` (Pydantic models in `src/FORMA/core/config/`).

Key groups:

- **LLM**: `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`, `LLM_TEMPERATURE`, `LLM_MAX_TOKENS`, `LLM_THINKING` (`enabled`/`disabled`/`none`; vendor-specific thinking payloads are built in `core/llm.py`)
- **I/O**: `RUN_MODE` (`s`=single / `b`=batch), `INPUT_DIR`, `OUTPUT_DIR`, `FILE_NAME`, batch range `FILE_BATCH_HEADER/START/END`
- **Redrock**: `REDROCK`, `RR_TEMPLATE_DIR`, `ARCHETYPE_DIR`, `USE_ARCHETYPES`, `NMINIMA`, `NNEAREST`, `OMP_NUM_THREADS`
- **Pipeline**: `HARNESS_CONCURRENCY` (parallel hypotheses), `MAX_TRIES`, `RETRY_DELAY`, camera arms (`ARM_NAME`, `ARM_WAVELENGTH_RANGE`), CWT thresholds (`CWT_*`), self-evolve (`SELF_EVOLVE`, `EXPECTED_Z`, `EXPECTED_TYPE`, …)

**Never commit `.env`** — it contains API keys and is git-ignored.

## Build and Run Commands

### Local (recommended for development)

```bash
pip install -r requirements.txt
pip install -e .            # or: pip install -e ".[web]" for the WebUI (gradio)
cp .env_example .env        # then fill in LLM_* and paths

# Redrock is required when REDROCK=true (external subprocess, needs templates):
git clone https://github.com/desihub/redrock && cd redrock
git clone https://github.com/desihub/redrock-templates py/redrock/templates
pip install -e . && pip install desiutil desispec
# optional archetypes: git clone https://github.com/desihub/redrock-archetypes

python scripts/main.py      # run the pipeline
python scripts/webui.py     # Gradio WebUI on :7860
```

### Docker

```bash
# CLI (single file or batch; .env is mounted via env_file)
INPUT_DIR_HOST=/path/to/fits OUTPUT_DIR_HOST=/path/to/results \
  docker compose run -e FILE_NAME=QSO_116 -e RUN_MODE=s forma-cli

# WebUI → http://localhost:7860
docker compose up forma-web
```

- First build takes 5–10 min (Redrock C-extension compilation). If GitHub is unreachable from Docker, use `REDROCK_SKIP_GIT_CLONE=true` + `REDROCK_SOURCE_DIR` (see Dockerfile header comments).
- `docker-compose.dev.yml` is auto-merged when running compose from the repo root and bind-mounts `src/FORMA` and `scripts/` for live iteration.
- Entrypoint modes (`docker/entrypoint.sh`): `cli` (default), `web`, `bash`.

### Outputs

Per input `{name}.fits`, results land in `{OUTPUT_DIR}/{name}/`: `final_report.md`, `{name}_in_brief.json`, plus subdirs `visual_interpreter/`, `single_hypothesis/`, `hypothesis_synthesis/`, `feature_auditor/`, `result_auditor/`, `report_writer/`, `{name}_redrock/`. Batch runs also append a summary row per file to `{OUTPUT_DIR}/in_brief.csv`. See README.md "Output Files" for the full tree.

## Testing

- **There is no automated test suite** — no pytest/unittest configuration, and `package.json`'s test script is a placeholder. Validation is done by running the pipeline against `example/` spectra and comparing with known DESI Visual Inspection (VI) results stored in each FITS `FIBERMAP` HDU (`VI_Z`, `VI_SPECTYPE`, `VI_QUALITY`).
- Quick smoke check after a change: run single mode on a known example, e.g. `FILE_NAME=QSO_116` with `INPUT_DIR` pointing at `example/basic/`, and inspect `{OUTPUT_DIR}/QSO_116/final_report.md` and `*_in_brief.json`.
- `example/counter_fact_examples/` contains forged spectra (Lyα removed; narrow-line impostor) used to stress-test the Result Auditor in isolation — see its README before modifying auditor logic.
- The self-evolve pathway (`SELF_EVOLVE=true` with `EXPECTED_Z`/`EXPECTED_TYPE`) is the closest thing to a regression harness: it records mismatches against ground truth and runs batch root-cause analysis.

## Code Style and Conventions

- **Language**: Python 3.12+, fully `async` (asyncio + LangGraph `ainvoke`); comments and docstrings are **bilingual Chinese/English** — match the surrounding file's style. User-facing docs (README, AGENTS.md) are in English; `README_Chinese.md` is the Chinese mirror.
- **Imports**: absolute imports under the `FORMA` package (`from FORMA.core...`). `scripts/` insert `src/` into `sys.path` directly, so scripts also work without installing the package.
- **Config**: never read `os.getenv` scattered through agent code — add fields to the appropriate Pydantic model in `src/FORMA/core/config/` and document the variable in `.env_example` (bilingual comments).
- **State**: extend `SpectroState` for inter-agent data; leading-underscore keys (e.g. `_no_features`, `_failure_recorded`) are internal control flags.
- **Agent pattern**: agents subclass `BaseAgent` (`agents/common/base_agent.py`), get their LLM lazily via `RuntimeContainer.get_model()`, and persist artifacts through `ResultWriter` into the per-file output directory. LLM-facing prompts live as Markdown "skills" in `agents/multi_agents/harness/skills/<AgentName>/` and domain knowledge in `harness/kb/` — update those when changing agent behavior.
- **Vendor quirks**: `core/llm.py` contains deliberate API-compatibility patches (DeepSeek `max_tokens` vs `max_completion_tokens`; thinking-mode `extra_body` per vendor). Preserve these when touching LLM client creation.
- **Retry semantics**: list-valued state fields can accumulate duplicates across retries; the orchestrator resets to a clean deep copy on connection/timeout errors — keep this invariant when adding retryable stages.
- **Minimal changes**: several features are intentionally commented out rather than deleted (PNG/OCR spectrum-extraction path, MCP deps in requirements.txt, brute-force line matching). Treat dated comment blocks (e.g. `# 已注释（2026-07-06）`) as deliberate history, not dead code to clean up.

## Security Considerations

- `.env` holds LLM API keys — it is git-ignored; never commit it, log it, or echo it. Use `.env_example` for new variables.
- The pipeline runs external subprocesses (Redrock `rrdesi`) and makes outbound HTTPS calls to the configured LLM endpoint; inputs are local FITS files.
- Docker containers run as `${UID}:${GID}` (non-root) and mount the input directory read-only; keep it that way.
- `data/`, `test_set/`, `log/`, `.paper/`, `.backup/`, `.debugger/`, `.claude/`, `.agents/` are git-ignored working directories — do not rely on their contents being present in a fresh clone, and do not commit generated outputs.
