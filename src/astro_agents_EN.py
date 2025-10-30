# English version of astro_agents.py

import json
import os
import logging
import traceback
from typing import Any

from .context_manager import SpectroState
from .base_agent import BaseAgent
from .mcp_manager import MCPManager

from .utils import (
    _detect_axis_ticks, _detect_chart_border, _crop_img,
    _remap_to_cropped_canvas, _pixel_tickvalue_fitting,
    _process_and_extract_curve_points, _convert_to_spectrum,
    _find_features_multiscale, _plot_spectrum, _plot_features,
    parse_list, getenv_float, getenv_int
)

# 配置 logger
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# 1. Visual Assistant — 负责图像理解与坐标阅读
# ---------------------------------------------------------

class SpectralVisualInterpreter(BaseAgent):
    """
    SpectralVisualInterpreter:
    Automatically extract information such as axis ticks, borders, pixel-to-value mappings and peaks/troughs 
    from input scientific spectral plots.
    """

    def __init__(self, mcp_manager: MCPManager):
        super().__init__(
            agent_name='Spectral Visual Interpreter',
            mcp_manager=mcp_manager
        )

    # --------------------------
    # Step 1.1: detect axis tick marks
    # --------------------------
    async def detect_axis_ticks(self, state: SpectroState):
        """
        Call the visual LLM to detect axis tick marks. 
        Raise an error if no image is provided or if the image is not a spectrum plot.
        """
        try:
            if not state['image_path'] or not os.path.exists(state['image_path']):
                raise ValueError("No image input or image path does not exist")

            prompt = """
You are a professional visual analysis model, specialized in extracting axis tick information from scientific charts.
If the input does not contain a spectrum plot, output "Not a spectrum plot".
Strictly output according to the following JSON Schema:
{
  "x_axis": {
    "label_and_Unit": "str",
    "tick_range": {"min": float, "max": float},
    "ticks": List[float]
  },
  "y_axis": {
    "label_and_Unit": "str",
    "tick_range": {"min": float, "max": float},
    "ticks": List[float]
  }
}
"""

            axis_info = await self.call_llm_with_context(
                prompt,
                image_path=state['image_path'],
                parse_json=True,
                description="axis info"
            )

            if axis_info == "Not a spectrum plot":
                raise ValueError("The image is not a spectrum plot")

            state["axis_info"] = axis_info
        
        except Exception as e:
            logger.error("❌ [detect_axis_ticks] Failed to detect axis ticks", exc_info=True)
            raise RuntimeError(f"Axis tick detection failed: {str(e)}") from e

    # --------------------------
    # Step 1.3: Merge the visual LLM and OCR results
    # --------------------------
    async def combine_axis_mapping(self, state: SpectroState):
        """Generate pixel-to-value mapping by combining visual LLM and OCR results"""
        try:
            axis_info_json = json.dumps(state['axis_info'], ensure_ascii=False)
            ocr_json = json.dumps(state['OCR_detected_ticks'], ensure_ascii=False)

            prompt_1 = f"""
You are a **scientific chart interpretation assistant**.

You will receive two sets of axis tick information:

1. **Visual model output:** 
{axis_info_json}
2. **OCR/OpenCV output:** 
{ocr_json}

---

### **Task:**

- According to those results above, perform **consistency correction and completion** on the OCR tick recognition data.

---

### **Rules:**

1. **Conflict Correction**
   - If conflicts exist between the OCR and visual model results (e.g., mismatched positions or values), prioritize the visual model results and correct the OCR data accordingly.
2. **Monotonicity Constraints**
   - For the **x-axis**, `position_x` must increase monotonically as the tick values increase.
   - For the **y-axis**, `position_y` must decrease monotonically as the tick values increase.
3. **Missing Data Completion**
   - If tick values are missing, first attempt linear interpolation using adjacent ticks.
   - If interpolation is not possible (e.g., at boundaries), fill the value with `null`.
   - If `bounding-box-scale_x` or `bounding-box-scale_y` are missing, fill them with `null`.
4. **Derived Parameter Calculation**
   - Compute `sigma_pixel = bounding-box-scale / 2`.
   - If the corresponding scale is missing, set `sigma_pixel` to `null`.
5. **Confidence Assignment (`conf_llm`)**
   - High-confidence OCR result → **0.9**
   - Interpolated or corrected result → **0.7**
   - Missing value inferred from visual prediction → **0.5** 

"""
            prompt_2 = """
Output:  
- Strictly output a JSON array, each element containing: 
{
  "axis" ("x" or "y"): "str", 
  "value": float, 
  "position_x": int, 
  "position_y": int,
  "bounding-box-scale_x": int, 
  "bounding-box-scale_y": int, 
  "sigma_pixel": float, 
  "conf_llm": float
}  
- Do not output any explanations or additional text.
"""
            prompt = prompt_1 + prompt_2
            tick_pixel_raw = await self.call_llm_with_context(
                prompt,
                # image_path=state['image_path'],
                parse_json=True,
                description="tick-value-to-pixel mapping"
            )

            state["tick_pixel_raw"] = tick_pixel_raw
        except Exception as e:
            logger.error("❌ [combine_axis_mapping] Failed to combine axis mapping", exc_info=True)
            raise RuntimeError(f"Axis mapping combination failed: {str(e)}") from e

    # --------------------------
    # Step 1.4: Check and correct
    # --------------------------
    async def revise_axis_mapping(self, state: SpectroState):
        """Check and correct the mapping between tick values and pixel positions."""
        try: 
            axis_mapping_json = json.dumps(state['tick_pixel_raw'], ensure_ascii=False)

            prompt = f"""
You are a scientific chart reading assistant.
Check the following mapping between tick values and pixel positions:
{axis_mapping_json}

Rules:

- Y-axis: pixel positions must strictly decrease as values increase
- X-axis: pixel positions must strictly increase as values increase
- null values are allowed

If there are issues, revise and output the JSON; otherwise, return the original input.
Do not output any explanations or extra text.
"""
            tick_pixel_revised = await self.call_llm_with_context(
                prompt,
                # image_path=state['image_path'],
                parse_json=True,
                description="revised tick-value-to-pixel mapping"
            )

            state["tick_pixel_raw"] = tick_pixel_revised
        except Exception as e:
            logger.error("❌ [revise_axis_mapping] Failed to revise axis mapping", exc_info=True)
            raise RuntimeError(f"Axis mapping revision failed: {str(e)}") from e

    def _load_feature_params(self):
        """Safely retrieve the parameters for peak and trough detection"""
        sigma_list = parse_list(os.getenv("SIGMA_LIST"), [2, 4, 16])
        tol_pixels = getenv_int("TOL_PIXELS", 10)
        prom_peaks = getenv_float("PROM_THRESHOLD_PEAKS", 0.01)
        prom_troughs = getenv_float("PROM_THRESHOLD_TROUGHS", 0.05)
        weight_original = getenv_float("WEIGHT_ORIGINAL", 1.0)
        plot_peaks = getenv_int("PLOT_PEAKS_NUMBER", 10)
        plot_troughs = getenv_int("PLOT_TROUGHS_NUMBER", 15)

        return sigma_list, tol_pixels, prom_peaks, prom_troughs, weight_original, plot_peaks, plot_troughs

    # --------------------------
    # Step 1.1~1.11: main
    # --------------------------
    async def run(self, state: SpectroState, plot: bool = True):
        """Run the complete visual analysis pipeline with per-step error reporting."""
        step_errors = []

        try:
            await self.detect_axis_ticks(state)
            logger.info("✅ Step 1.1: Axis ticks detected")

            state["OCR_detected_ticks"] = _detect_axis_ticks(state['image_path'])
            logger.info("✅ Step 1.2: OCR ticks extracted")

            await self.combine_axis_mapping(state)
            logger.info("✅ Step 1.3: Axis mapping combined")

            await self.revise_axis_mapping(state)
            logger.info("✅ Step 1.4: Axis mapping revised")

            state["chart_border"] = _detect_chart_border(state['image_path'])
            _crop_img(state['image_path'], state["chart_border"], state['crop_path'])
            logger.info("✅ Step 1.5: Image cropped")

            state["tick_pixel_remap"] = _remap_to_cropped_canvas(state['tick_pixel_raw'], state["chart_border"])
            logger.info("✅ Step 1.6: Pixel remapping completed")

            state["pixel_to_value"] = _pixel_tickvalue_fitting(state['tick_pixel_remap'])
            logger.info("✅ Step 1.7: Pixel-to-value mapping fitted")

            curve_points, curve_gray_values = _process_and_extract_curve_points(state['crop_path'])
            state["curve_points"] = curve_points
            state["curve_gray_values"] = curve_gray_values
            logger.info("✅ Step 1.8: Curve points extracted")

            state["spectrum"] = _convert_to_spectrum(state['curve_points'], state['curve_gray_values'], state['pixel_to_value'])
            logger.info("✅ Step 1.9: Spectrum converted")

            # Step 1.10
            sigma_list, tol_pixels, prom_peaks, prom_troughs, weight_original, plot_peaks, plot_troughs = self._load_feature_params()
            state['sigma_list'] = sigma_list
            try:
                state["peaks"] = _find_features_multiscale(
                    state, feature="peak", sigma_list=sigma_list,
                    prom=prom_peaks, tol_pixels=tol_pixels, weight_original=weight_original,
                    use_continuum_for_trough=True
                )
                state["troughs"] = _find_features_multiscale(
                    state, feature="trough", sigma_list=sigma_list,
                    prom=prom_troughs, tol_pixels=tol_pixels, weight_original=weight_original,
                    use_continuum_for_trough=True,
                    min_depth=0.08
                )
                logger.info(f"✅ Step 1.10: Detected {len(state['peaks'])} peaks and {len(state['troughs'])} troughs")
            except Exception as e:
                logger.exception("❌ Step 1.10: Feature detection failed")
                raise

            # Step 1.11: Plotting
            if plot:
                try:
                    state["spectrum_fig"] = _plot_spectrum(state)
                    state["features_fig"] = _plot_features(state, sigma_list, [plot_peaks, plot_troughs])
                    logger.info("✅ Step 1.11: Plots generated")
                except Exception as e:
                    logger.exception("❌ Step 1.11: Plotting failed")
                    raise

            return state
        
        except Exception as e:
            print(f"❌ run pipeline terminated with error: {e}")
            raise

# ---------------------------------------------------------
# 2. Rule-based Analyst — 负责基于规则的物理分析
# ---------------------------------------------------------
class SpectralRuleAnalyst(BaseAgent):
    
    """
    Rule-based analyst: 
    perform qualitative analysis based on given physical and spectral line knowledge.
    """

    def __init__(self, mcp_manager: MCPManager):
        super().__init__(
            agent_name='Spectral Rule Analyst',
            mcp_manager=mcp_manager
        )

    async def describe_spectrum_picture(self, state: SpectroState):
        prompt = f"""
You are an experienced astronomical spectral analysis assistant.

You will be presented with an astronomical spectrum (from an object with unknown redshift).

Using the image, **qualitatively describe the overall morphology of the spectrum**, including but not limited to the following aspects:

---

### Step 1: Continuum Shape

- Overall flux distribution trend (e.g., enhanced at blue end / enhanced at red end / roughly flat / arch-shaped, etc.).
- Whether features of a power-law continuum, blackbody continuum, or flat continuum are apparent.
- Any obvious breaks or inflection points in the continuum (e.g., Balmer break, Lyα forest region, etc.).

### Step 2: Major Emission and Absorption Features

- Presence of prominent emission peaks or absorption troughs.
- Approximate number and relative strengths of emission or absorption lines.
- Whether these lines are broad or narrow, symmetric or asymmetric.
- Avoid giving precise numerical values (exact wavelengths or fluxes); just describe relative positions and characteristics.

### Step 3: Overall Structure and Noise Features

- General impression of the spectrum’s signal-to-noise ratio (high / medium / low).
- Presence of noise fluctuations, abnormal spikes, or data gaps.
- Any change in data quality toward the short-wavelength or long-wavelength ends.

---

⚠️ **Notes:**

- Do not output precise numerical values or tables.
- Do not attempt to calculate redshift.
- Focus on visual and morphological description, making qualitative judgments like a human astronomer.
- Do not call any external tools.

Finally, present your observations in a structured format, for example, using section headings:

- (Continuum)
- (Emission & Absorption)
- (Noise & Data Quality)
"""
        
        response = await self.call_llm_with_context(
            prompt,
            image_path=state['image_path'],
            parse_json=False,
            description="Visual Qualitative Description of a Spectrum"
        )
        state['visual_interpretation'] = response
        
    
    async def preliminary_classification(self, state: SpectroState) -> str:
        """
        Preliminary Classification: 
        Make an initial assessment of the object type based on the spectral morphology.
        """

        visual_interpretation_json = json.dumps(state['visual_interpretation'], ensure_ascii=False)
        prompt = f"""
You are an experienced astronomical spectral analysis assistant.

You will see an astronomical spectrum (from an object with unknown redshift), which may belong to one of the following three categories:

- **Star**: Strong continuum, spectral lines are usually absorption lines (e.g., Balmer series, metal lines), with little or no noticeable redshift.
- **Galaxy**: Exhibits some redshift, often with emission or absorption lines (e.g., [O II], Hβ, [O III], Hα), lines are relatively narrow, and the continuum is relatively weak.
- **QSO**: Strong broad emission lines spanning the visible/UV range, line widths significantly larger than typical galaxies, usually with noticeable redshift.

The previous astronomical assistant has already provided a qualitative description of the overall spectral morphology:

{visual_interpretation_json}

Based on their description, please assess which category or categories the spectrum is most likely to belong to, and provide a confidence level.

Your answer must strictly follow this format:

Guess 1:

- **Category**: Star / Galaxy / QSO (choose one)
- **Reason**: Brief explanation of the classification (e.g., line width, redshift features, continuum morphology)
- **Confidence**: High / Medium / Low

Guess 2:

- **Category**: Star / Galaxy / QSO (choose one)
- **Reason**: Brief explanation of the classification
- **Confidence**: High / Medium / Low

… and so on.

⚠️ **Note:**

- Only provide answers with medium confidence or higher
- Do not include exact numerical values or tables
- Do not attempt to calculate redshift
- Focus on visual and morphological assessment, making qualitative judgments like a human astronomer
- Do not call any external tools
"""
        response = await self.call_llm_with_context(
            prompt,
            image_path=state['image_path'],
            parse_json=False,
            description="Preliminary Classification"
        )
        state['preliminary_classification'] = response
        
    def _common_prompt_header_QSO(self, state, include_rule_analysis=True):
        visual_json = json.dumps(state['visual_interpretation'], ensure_ascii=False)
        peak_json = json.dumps(state['peaks'][:10], ensure_ascii=False)
        trough_json = json.dumps(state['troughs'], ensure_ascii=False)

        header = f"""
You are an astronomical spectral analysis assistant.

The following information may come from a QSO spectrum with unknown redshift.

A previous assistant has already provided a preliminary description of this spectrum:
{visual_json}
"""

        if include_rule_analysis and state.get('rule_analysis'):
            rule_json = json.dumps("\n".join(str(item) for item in state['rule_analysis']), ensure_ascii=False)
            header += f"\nThe previous assistant also performed an initial analysis under the assumption that Lyα lines are present:\n{rule_json}\n"

        header += f"""
Using the original spectrum and Gaussian-smoothed curves with sigma={state['sigma_list']}, peaks and troughs were identified with scipy functions.
The discussion of peaks/troughs is based on the following data:
- Top 10 representative emission lines:
{peak_json}
- Possible absorption lines:
{trough_json}
"""
        return header

    def _common_prompt_tail(self, step_title, extra_notes=""):
        tail = f"""
---

Output format:
{step_title}
...

---

🧭 Notes:

- For computed (non-original) data, keep 3 decimal places in the output.
- No need for repetitive summaries.
- Do not repeat input data line by line.
- Focus on physical reasoning and reasonable explanations.
- Ensure the final output is complete; do not truncate midway.
"""
        if extra_notes:
            tail = extra_notes + "\n" + tail
        return tail
    
    async def step_1(self, state):
        header = self._common_prompt_header_QSO(state, include_rule_analysis=False)
        tail = self._common_prompt_tail("Step 1: Lyα Analysis")

        prompt = header + """
Analyze according to the following steps:

**Step 1: Lyα Line Detection**
Assuming the spectrum contains a Lyα emission line (λ_rest = 1216 Å):

1. Identify the most probable observed emission line corresponding to Lyα (select from the provided peak list).
2. Output:

   - **λ_obs** (observed wavelength)
   - **Intensity** (relative strength or qualitative description)
   - **Line width** (FWHM or approximate pixel width)
3. Use the tool `calculate_redshift` to compute the redshift (z) based on this emission line.
4. Examine the blue end (short-wavelength side) for Lyα forest features:

   - If absorption lines are denser, narrower, and clustered near the Lyα blue side, note this and provide a brief qualitative description.
""" + tail
        
        response = await self.call_llm_with_context(prompt, parse_json=False, description="Step 1 Lyα 分析")
        state['rule_analysis'].append(response)

    async def step_2(self, state):
        header = self._common_prompt_header_QSO(state)
        tail = self._common_prompt_tail("Step 2: Analysis of Other Significant Emission Lines")

        prompt = header + """
Continue the analysis:

**Step 2: Analysis of Other Significant Emission Lines**

1. Using the redshift obtained in Step 1 as a reference, use the tool `predict_obs_wavelength` to check whether other significant emission lines (e.g., C IV 1549, C III] 1909, Mg II 2799, Hβ, Hα, etc.) may be present in the spectrum. Do **not** calculate these manually.
2. Are there any other emission lines that require attention?
""" + tail

        response = await self.call_llm_with_context(prompt, parse_json=False, description="Step 2 Analysis of Other Significant Emission Lines")
        state['rule_analysis'].append(response)

    async def step_3(self, state):
        header = self._common_prompt_header_QSO(state)
        tail = self._common_prompt_tail("Step 3: Comprehensive Assessment")

        prompt = header + """
Continue the analysis:

**Step 3: Comprehensive Assessment**

- From Step 1 to Step 2, if there is insufficient evidence for the presence of Lyα (e.g., no clear peak at the corresponding wavelength or redshift inconsistent with other lines), **assume Lyα is absent** and terminate the analysis.
- Only include Lyα in the combined redshift calculation if there is strong evidence for its presence (significant peak + consistent redshift with other lines).
- If the redshift results from Step 1 and Step 2 are consistent, integrate the analyses from Step 1 and Step 2, using the matched lines to provide:

  - Redshift of each line
  - Weighted average redshift (z ± Δz), using the flux at the smallest shared sigma smoothing as the weight and the tool `weighted_average` (do **not** calculate manually)
  - The calculation of redshift must use the tool `calculate_redshift`; manual computation is not allowed.
- Provide the wavelengths and line names of all emission lines confirmed at this redshift.
""" + tail

        response = await self.call_llm_with_context(prompt, parse_json=False, description="Step 3 Comprehensive Assessment")
        state['rule_analysis'].append(response)

    async def step_4(self, state):
        header = self._common_prompt_header_QSO(state)
        tail = self._common_prompt_tail("Step 4: Supplementary Step (Suppose the strongest emission line is not Lyα)")

        prompt = header + """
Continue the analysis:

**Step 4: Supplementary Step (Suppose the strongest emission line is not Lyα)**

- Suppose the strongest emission line is not Lyα
- Based on typical QSO spectral features, identify the **strongest peak** in the spectrum.
- Guess which spectral line this peak may correspond to (e.g., C IV, C III], Mg II, Hβ, Hα, etc.).
- Follow the logic of Steps 1–3. Use the tool `calculate_redshift` for redshift calculations and `predict_obs_wavelength` for observed line wavelength predictions. Manual calculations are not allowed.

  - Output information for this peak’s spectral line:

    - Line name
    - λ_obs
    - Intensity
    - Line width
    - Preliminary redshift z based on λ_rest (must use `calculate_redshift`)
  - If possible, predict other visible emission lines and calculate their redshifts
  - Combine all lines to provide the most likely redshift and its range
- Does the evidence support the hypothesis that the strongest emission line is not Lyα?
""" + tail

        response = await self.call_llm_with_context(prompt, parse_json=False, description="Step 4: Supplementary Step (Suppose the strongest emission line is not Lyα)")
        state['rule_analysis'].append(response)

#     # --------------------------
#     # Run
#     # --------------------------
    async def run(self, state: SpectroState):
        try:
            await self.describe_spectrum_picture(state)
            await self.preliminary_classification(state)
            await self.step_1(state)
            await self.step_2(state)
            await self.step_3(state)
            await self.step_4(state)
            return state
        except Exception as e:
            import traceback
            print("❌ An error occurred during spectral analysis:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print("Full traceback:")
            traceback.print_exc()
            raise 
        



# # ---------------------------------------------------------
# # 3. Revision Supervisor — 负责交叉审核与评估
# # ---------------------------------------------------------
class SpectralAnalysisAuditor(BaseAgent):
    """
    Analysis Auditor: 
    audit and auditor the outputs of other analysis agents
    """

    def __init__(self, mcp_manager: MCPManager):
        super().__init__(
            agent_name='Spectral Analysis Auditor',
            mcp_manager=mcp_manager
        )

    def _common_prompt_header_QSO(self, state: SpectroState) -> str:
        peak_json = json.dumps(state['peaks'][:10], ensure_ascii=False)
        trough_json = json.dumps(state['troughs'], ensure_ascii=False)
        rule_analysis = "\n\n".join(str(item) for item in state['rule_analysis'])
        return f"""
You are a meticulous **Astronomical Spectrum Report Analysis Auditor**.

**Task Objectives:**

- audit the spectral analysis reports or conclusions from other analysts.
- Identify logical flaws, computational errors, inconsistencies, or incorrect inferences.
- Provide correction suggestions or additional analysis directions.

**Working Principles:**

- Maintain objectivity and critical thinking.
- Do not repeat the original analysis; only point out issues and suggest improvements.
- If the original report is reasonable, explicitly confirm its validity.
- Any calculations involving redshift or observed wavelengths must use the tools **calculate_redshift** and **predict_obs_wavelength**. Self-calculation is not allowed.

**Output Requirements:**

- Provide descriptive text.
- Concisely list audit comments (e.g., “Conclusion premature,” “Line interpretation correct”).
- For each issue identified, give a corresponding improvement suggestion.
- Provide an overall evaluation: Reliable / Partially Reliable / Unreliable.

**Known Data:**
The original spectrum and Gaussian-smoothed curves with sigma=2, sigma=4, and sigma=16 were used. Peaks and troughs were identified using SciPy functions. The peak/trough discussion is based on the following data:

- Representative top 10 emission lines:
  {peak_json}
- Potential absorption lines:
  {trough_json}

**Other Analyst’s Spectral Report:**

{rule_analysis}

The report retains 3 decimal places in redshift calculations.
"""

    async def auditing(self, state: SpectroState):
        header = self._common_prompt_header_QSO(state)

        if state['count'] == 0:
            body = f"""
Please audit this analysis report.
"""
        elif state['count']:     
            auditing_history = state['auditing_history'][-1]
            auditing_history_json = json.dumps(auditing_history, ensure_ascii=False)
            response_history = state['refine_history'][-1]
            response_history_json = json.dumps(response_history, ensure_ascii=False)

            body = f"""
Your latest concerns regarding this analysis report are:
{auditing_history_json}

The other analyst’s responses are:
{response_history_json}

Please reply to the other analyst’s responses and continue the audit.
"""
        prompt = header + body
        response = await self.call_llm_with_context(prompt, parse_json=False, description="Auditing")
        state['auditing_history'].append(response)

    async def run(self, state: SpectroState) -> SpectroState:
        await self.auditing(state)
        return state



# # ---------------------------------------------------------
# # 4. Refine Analyst — 自由回应审查并改进
# # ---------------------------------------------------------
class SpectralRefinementAssistant(BaseAgent):
    """Refinement Assistant: 
    Respond to the audit and improve the analysis.
    """

    def __init__(self, mcp_manager: MCPManager):
        super().__init__(
            agent_name='Spectral Refinement Assistant',
            mcp_manager=mcp_manager
        )

    def _common_prompt_header_QSO(self, state) -> str:
        peak_json = json.dumps(state['peaks'][:10], ensure_ascii=False)
        trough_json = json.dumps(state['troughs'], ensure_ascii=False)
        rule_analysis = "\n\n".join(str(item) for item in state['rule_analysis'])
        return f"""
You are a reflective **astronomical spectrum analyst**.

**Task objectives:**

- Read and understand other analysts’ spectral analysis reports.
- Read and comprehend feedback provided by the auditor.
- Refine your own or others’ prior analysis.
- Provide new explanations or revise conclusions.

**Working principles:**

- Carefully address each feedback point, explaining improvements one by one.
- If you consider the original conclusion correct, provide sufficient justification.
- Produce a final, more rigorous and complete analysis.
- All calculations of redshift and observed spectral wavelengths must use the tools `calculate_redshift` and `predict_obs_wavelength`. Manual calculations are not allowed.

**Output requirements:**

- Provide descriptive language.
- List the received feedback and your corresponding responses.
- Provide an improved spectral analysis summary.
- Explain the modifications made and their scientific rationale.

**Given:**

- Original spectra and Gaussian-smoothed curves with sigma = 2, 4, 16 have been processed using `scipy` for peak/trough detection.
- Discussion of peaks/troughs is based on the following data:

  - Representative top 10 emission lines:
    {peak_json}
  - Possible absorption lines:
    {trough_json}
- Other analysts’ spectral analysis report:
  {rule_analysis}

This report retained three decimal places for redshift calculations.
"""

    async def refine(self, state: SpectroState):
        header = self._common_prompt_header_QSO(state)
        auditing = state['auditing_history'][-1]
        auditing_json = json.dumps(auditing, ensure_ascii=False)
        body = f"""
The latest recommendations from the auditing analyst responsible for verifying the report are:
{auditing_json}

Please respond to these recommendations.
"""
        prompt = header + body
        response = await self.call_llm_with_context(prompt, parse_json=False, description="回应审查")
        state['refine_history'].append(response)

    async def run(self, state: SpectroState) -> SpectroState:
        try:
            await self.refine(state)
            return state
        except Exception as e:
            import traceback
            print("❌ An error occurred during spectral analysis:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print("Full traceback:")
            traceback.print_exc()
            # 可选：返回当前状态或抛出异常
            raise  # 如果你希望调用者也能捕获该异常


# ---------------------------------------------------------
# 🧩 5. Host Integrator — 汇总与总结多方观点
# ---------------------------------------------------------
class SpectralSynthesisHost(BaseAgent):
    """
    Synthesis Host: 
    Integrates analyses and conclusions from multiple agents
    """

    def __init__(self, mcp_manager: MCPManager):
        super().__init__(
            agent_name='Spectral Synthesis Host',
            mcp_manager=mcp_manager
        )

    def get_system_prompt(self) -> str:
        return f"""
You are a coordinating **Astronomical Spectral Analysis Host**.

**Task Objectives:**

- Summarize all outputs from the Visual Analyst, Rule Analyst, Auditor, and Re-Analyzer.
- Integrate conclusions from different perspectives to form the final spectral interpretation.
- Clearly highlight points of agreement and disagreement among the agents.

**Working Principles:**

- No need to call any tools.
- Do not blindly follow any single analysis.
- Maintain overall scientific accuracy and logical consistency.
- The final output must be traceable (indicate which agent provided the underlying evidence).

**Output Requirements:**

- Provide explanatory text.
- Keep numerical data to 3 decimal places.
- Only output the analysis content; no need to declare the source of each section.
- Provide the final integrated conclusion and confidence rating (high/medium/low).
- Explicitly indicate if uncertainty remains.
- Follow the specified format; do not include any extraneous content.
"""


    async def summary(self, state) -> str:
        try:
            preliminary_classification_json = json.dumps(state['preliminary_classification'], ensure_ascii=False)
            visual_interpretation_json = json.dumps(state['visual_interpretation'], ensure_ascii=False)
            rule_analysis = "\n\n".join(str(item) for item in state['rule_analysis'])
            rule_analysis_json = json.dumps(rule_analysis, ensure_ascii=False)
            auditing = "\n\n".join(str(item) for item in state['auditing_history'])
            auditing_json = json.dumps(auditing, ensure_ascii=False)
            refine = "\n\n".join(str(item) for item in state['refine_history'])
            refine_json = json.dumps(refine, ensure_ascii=False)
        except Exception as e:
            print("❌ An error occurred during spectral analysis:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            raise

        header = self.get_system_prompt()

        prompt = f"""

Visual description of the spectrum:
{visual_interpretation_json}

preliminary classification of the spectrum type: 
{preliminary_classification_json}

Rule Analyst's perspective:
{rule_analysis_json}

Auditor's perspective:
{auditing_json}

Refining Analyst's perspective:
{refine_json}

**Output format:**

- **Visual characteristics of the spectrum**
- **Analysis report** (integrate the perspectives of the Rule Analyst, Auditor, and Refining Analyst, structured step by step)

  - Step 1
  - Step 2
  - Step 3
  - Step 4
- **Conclusion**

  - Object type and redshift (z ± Δz)
  - Verified spectral lines (output as Line Name - λ_rest - λ_obs)
  - Signal-to-noise ratio of the spectrum
  - Confidence rating of the analysis report (if ≥2 lines verified: “high”; if 1 line verified: “medium”; otherwise: “low”)
  - Whether manual intervention is needed
"""
        prompt = header + prompt
        response = await self.call_llm_with_context(prompt, parse_json=False, description="总结")
        state['summary'] = response

    async def in_brief(self, state):
        summary_json = json.dumps(state['summary'], ensure_ascii=False)
        prompt_type = f"""
You are an Astronomy Spectrum Analysis Host responsible for coordinating the final synthesis.

You have already produced a summary of an astronomical spectrum:
{summary_json}

- Please output the **object type** from the **Conclusion** section (Choose between these 3 words: star, galaxy, QSO).

- Output format: str  
- Do not output any other information.
"""
        response_type = await self.call_llm_with_context(prompt_type, parse_json=False, description="Summary")
        state['in_brief']['type'] = response_type

        prompt_redshift = f"""
You are an Astronomy Spectrum Analysis Host responsible for coordinating the final synthesis.

You have already produced a summary of an astronomical spectrum:
{summary_json}

Please output the **redshift z** from the **Conclusion** section (no ± Δz).

- Output format: float  
- Do not output any other information.
"""
        response_redshift = await self.call_llm_with_context(prompt_redshift, parse_json=False, description="Summary")
        state['in_brief']['redshift'] = response_redshift

        prompt_rms = f"""
You are an Astronomy Spectrum Analysis Host responsible for coordinating the final synthesis.

You have already produced a summary of an astronomical spectrum:
{summary_json}

Please output the **redshift uncertainty Δz** from the **Conclusion** section (do not output z).

- Output format: float  
- Do not output any other information.
"""
        response_rms = await self.call_llm_with_context(prompt_rms, parse_json=False, description="Summary")
        state['in_brief']['rms'] = response_rms

    async def run(self, state: SpectroState) -> SpectroState:
        try:
            await self.summary(state)
            await self.in_brief(state)
            return state
        except Exception as e:
            import traceback
            print("❌ An error occurred during spectral analysis:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print("Full traceback:")
            traceback.print_exc()