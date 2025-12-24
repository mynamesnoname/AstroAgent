import json
import os
import numpy as np
import matplotlib.pyplot as plt
import logging

from scipy.ndimage import gaussian_filter1d

from .context_manager import SpectroState
from .base_agent import BaseAgent
from .mcp_manager import MCPManager

from .utils import (
    _detect_chart_border, _crop_img, _remap_to_cropped_canvas, 
    _pixel_tickvalue_fitting, _process_and_extract_curve_points, _convert_to_spectrum,
    _find_features_multiscale, _plot_spectrum, getenv_int, 
    _load_feature_params, merge_features, plot_cleaned_features, 
    safe_to_bool, find_overlap_regions, _detect_axis_ticks_tesseract,
    _detect_axis_ticks_paddle
)

# ---------------------------------------------------------
# 1. Visual Assistant — Responsible for image understanding and coordinate reading
# ---------------------------------------------------------
class SpectralVisualInterpreter(BaseAgent):
    """
    SpectralVisualInterpreter

    Automatically extracts axis ticks, borders, pixel mappings, peaks/troughs, and other information from scientific spectral plots.
    """
    
    def __init__(self, mcp_manager: MCPManager):
        super().__init__(
            agent_name='Spectral Visual Interpreter',
            mcp_manager=mcp_manager
        )

    # --------------------------
    # Step 1.1: Detect axis ticks
    # --------------------------
    async def detect_axis_ticks(self, state: SpectroState):
        """
        Invoke a vision LLM to detect axis ticks. Raise an error if no image is provided or the image is not a spectral plot.
        """
        class NoImageError(Exception): pass
        class NotSpectralImageError(Exception): pass

        if not state['image_path'] or not os.path.exists(state['image_path']):
            print(state['image_path'])
            raise NoImageError("❌ No image provided or image path does not exist")

        system_prompt = state['prompt'][f'{self.agent_name}']['detect_axis_ticks']['system_prompt']
        user_prompt = state['prompt'][f'{self.agent_name}']['detect_axis_ticks']['user_prompt']

        axis_info = await self.call_llm_with_context(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_path=state['image_path'],
            parse_json=True,
            description="Axis information"
        )
        if axis_info == "Non-spectral image":
            raise NotSpectralImageError(f"❌ The input image is not a spectral plot. LLM output: {axis_info}")
        return axis_info

    # --------------------------
    # Steps 1.2–1.3: Merge visual + OCR ticks
    # --------------------------
    async def combine_axis_mapping(self, state: SpectroState):
        """Combine vision-based results and OCR results to generate a pixel-to-value mapping."""
        try:
            axis_info_json = json.dumps(state['axis_info'], ensure_ascii=False)
            ocr_json = json.dumps(state['OCR_detected_ticks'], ensure_ascii=False)

            system_prompt = state['prompt'][f'{self.agent_name}']['combine_axis_mapping']['system_prompt']
            user_prompt = state['prompt'][f'{self.agent_name}']['combine_axis_mapping']['user_prompt'].format(
                axis_info_json=axis_info_json,
                ocr_json=ocr_json
            )
            tick_pixel_raw = await self.call_llm_with_context(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                image_path=None,
                parse_json=True,
                description="Tick-to-pixel mapping"
            )
            return tick_pixel_raw
            # print(tick_pixel_raw)
        except Exception as e:
            logging.error(f"Error in combine_axis_mapping: {e}")
            raise e

    # --------------------------
    # Step 1.4: Validation and correction
    # --------------------------
    async def revise_axis_mapping(self, state: SpectroState):
        """Verify and correct the correspondence between tick values and pixel positions."""
        try:
            axis_mapping_json = json.dumps(state['tick_pixel_raw'], ensure_ascii=False)

            system_prompt = state['prompt'][f'{self.agent_name}']['revise_axis_mapping']['system_prompt']
            user_prompt = state['prompt'][f'{self.agent_name}']['revise_axis_mapping']['user_prompt'].format(
                axis_mapping_json=axis_mapping_json
            )

            tick_pixel_revised = await self.call_llm_with_context(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                image_path=None,
                parse_json=True,
                description="Revised tick mapping"
            )
            return tick_pixel_revised
        except Exception as e:
            logging.error(f"Error in revise_axis_mapping: {e}")
            raise e

    # --------------------------
    # Step 1.5: Image cropping
    # --------------------------
    async def check_border(self, state):
        try:
            system_prompt = """
You are a professional scientific chart analysis assistant specialized in processing matplotlib-style spectral plots in astronomy. You can accurately identify whether residual axis borders or decorative lines remain along the image edges and make precise judgments based on visual content.
"""
            user_prompt = """
You will receive two images:
- One is the original spectral image, which may include plot borders.
- The other is a preprocessed matplotlib astronomical spectral plot after OCR and OpenCV operations, where an attempt has already been made to crop out the original chart's borders and external regions.

Please evaluate whether obvious straight-line border remnants (e.g., long, straight black or dark line segments—typically part of the outer axis frame) remain along each of the four edges (top, right, bottom, left).

Judgment criteria:
- If **no such line segment** is visible on a given edge, mark it as “cleanly cropped”.
- If **any clear line segment** (even very thin) remains visible on an edge, mark it as “not cleanly cropped”.

Output your result strictly in the following JSON format, containing only four keys. Values must be the string 'true' (clean) or 'false' (not clean):

{
    "top": "true" or "false",
    "right": "true" or "false",
    "bottom": "true" or "false",
    "left": "true" or "false"
}

Do not output any additional content.
"""
            response = await self.call_llm_with_context(
                system_prompt,
                user_prompt,
                image_path=[state['image_path'], state['crop_path']],
                parse_json=True,
                description='Border cropping check'
            )
            response['top'] = safe_to_bool(response['top'])
            response['right'] = safe_to_bool(response['right'])
            response['bottom'] = safe_to_bool(response['bottom'])
            response['left'] = safe_to_bool(response['left'])
            return response
        except:
            logging.error(f"Error in check_border: {response}")

    async def peak_trough_detection(self, state: SpectroState):
        try:
            sigma_list, tol_pixels, prom_peaks, prom_troughs, _, _ = _load_feature_params()
            state['sigma_list'] = sigma_list

            spec = state["spectrum"]
            wavelengths = np.array(spec["new_wavelength"])
            flux = np.array(spec["weighted_flux"])

            state["peaks"] = _find_features_multiscale(
                wavelengths, flux,
                state, feature="peak", sigma_list=sigma_list,
                prom=prom_peaks, tol_pixels=tol_pixels,
                use_continuum_for_trough=True
            )
            state["troughs"] = _find_features_multiscale(
                wavelengths, flux,
                state, feature="trough", sigma_list=sigma_list,
                prom=prom_troughs, tol_pixels=tol_pixels, 
                use_continuum_for_trough=True,
                min_depth=0.08
            )
            # print(len(state["peaks"]), len(state["troughs"]))

            # Divide wavelengths into ROIs of 500 Ångströms each and perform peak/trough detection per ROI
            ROI_peaks = []
            ROI_troughs = []
            roi_size = 500  # Width of each ROI in Ångströms
            roi_edges = np.arange(wavelengths[0], wavelengths[-1], roi_size)
            for i in range(len(roi_edges)-1):
                roi_start = roi_edges[i]
                roi_end = roi_edges[i+1]
                mask = (wavelengths >= roi_start) & (wavelengths < roi_end)
                roi_wavelengths = np.where(mask, wavelengths, 0)
                roi_flux = np.where(mask, flux, 0)
                if len(roi_wavelengths) == 0:
                    continue
                roi_peaks = _find_features_multiscale(
                    roi_wavelengths, roi_flux,
                    state, feature="peak", sigma_list=sigma_list,
                    prom=prom_peaks, tol_pixels=tol_pixels,
                    use_continuum_for_trough=True
                )
                roi_troughs = _find_features_multiscale(
                    roi_wavelengths, roi_flux,
                    state, feature="trough", sigma_list=sigma_list,
                    prom=prom_troughs, tol_pixels=tol_pixels,
                    use_continuum_for_trough=True,
                    min_depth=0.08
                )
                ROI_peaks.extend(roi_peaks)
                ROI_troughs.extend(roi_troughs)
            roi_edges_ = roi_edges + 250
            for i in range(len(roi_edges_)-1):
                roi_start = roi_edges_[i]
                roi_end = roi_edges_[i+1]
                mask = (wavelengths >= roi_start) & (wavelengths < roi_end)
                roi_wavelengths = np.where(mask, wavelengths, 0)
                roi_flux = np.where(mask, flux, 0)
                if len(roi_wavelengths) == 0:
                    continue
                roi_peaks = _find_features_multiscale(
                    roi_wavelengths, roi_flux,
                    state, feature="peak", sigma_list=sigma_list,
                    prom=prom_peaks, tol_pixels=tol_pixels,
                    use_continuum_for_trough=True
                )
                roi_troughs = _find_features_multiscale(
                    roi_wavelengths, roi_flux,
                    state, feature="trough", sigma_list=sigma_list,
                    prom=prom_troughs, tol_pixels=tol_pixels,
                    use_continuum_for_trough=True,
                    min_depth=0.08
                )
                ROI_peaks.extend(roi_peaks)
                ROI_troughs.extend(roi_troughs)
            state["ROI_peaks"] = ROI_peaks
            state["ROI_troughs"] = ROI_troughs
            state['merged_peaks'], state['merged_troughs'] = merge_features(
                wavelengths, flux,
                global_peaks=state["peaks"],
                global_troughs=state["troughs"],
                ROI_peaks=state["ROI_peaks"],
                ROI_troughs=state["ROI_troughs"],
                tol_pixels=tol_pixels
            )
        except Exception as e:
            print(f"Error in peak_trough_detection: {e}")
        return state

    async def continuum_fitting(self, state: SpectroState):
        """Perform simple continuum fitting."""
        try:
            spec = state["spectrum"]
            wavelengths = np.array(spec["new_wavelength"])
            flux = np.array(spec["weighted_flux"])

            band_name = state['band_name']
            band_wavelength = state['band_wavelength']
            print('cut continuum')
            if band_name:
                overlap_regions = find_overlap_regions(band_name, band_wavelength)
                # 初始化 mask 为全 False
                mask = np.zeros_like(wavelengths, dtype=bool)
                for key in overlap_regions:
                    low, high = overlap_regions[key]
                    region_mask = (wavelengths >= low) & (wavelengths <= high)
                    mask = mask | region_mask  # 或者用 mask |= region_mask
                wavelengths = wavelengths[~mask]
                flux = flux[~mask]
                
            sigma_contunuum = getenv_int('CONTINUUM_SMOOTHING_SIGMA', None)
            print(f'CONTINUUM_SMOOTHING_SIGMA: {sigma_contunuum}')
            if sigma_contunuum is None:
                logging.error("CONTINUUM_SMOOTHING_SIGMA is not set")
                return
            continuum_flux = gaussian_filter1d(flux, sigma=sigma_contunuum)
            state['continuum'] = {
                'wavelength': wavelengths.tolist(),
                'flux': continuum_flux.tolist()
            }
        except Exception as e:
            print(f"Error in continuum_fitting: {e}")
        return state

    # --------------------------
    # Steps 1.1–1.11: Main pipeline
    # --------------------------
    async def run(self, state: SpectroState, plot: bool = True):
        """Execute the full visual analysis pipeline."""
        try:
            # Step 1.1: Use vision LLM to extract axis info
            state["axis_info"] = await self.detect_axis_ticks(state)
            # Step 1.2: Extract ticks via OCR
            OCR = os.getenv('OCR', 'paddle')
            print(f"OCR: {OCR}")
            if OCR == 'paddle':
                state['OCR_detected_ticks'] = _detect_axis_ticks_paddle(state)
            else:
                state['OCR_detected_ticks'] = _detect_axis_ticks_tesseract(state)
            # print(state["OCR_detected_ticks"])
            # Step 1.3: Merge results
            state["tick_pixel_raw"] = await self.combine_axis_mapping(state)
            # Step 1.4: Revise mapping
            state["tick_pixel_raw"] = await self.revise_axis_mapping(state)
            # Step 1.5: Border detection and cropping
            state['margin'] = {
                'top': 20,
                'right': 10,
                'bottom': 15,
                'left': 10,
            }
            stop = False
            while not stop:
                state["chart_border"] = _detect_chart_border(state['image_path'], state['margin'])
                _crop_img(state['image_path'], state["chart_border"], state['crop_path'])
                box_new = await self.check_border(state)
                values = [box_new['top'], box_new['bottom'], box_new['left'], box_new['right']]
                margin = [state['margin']['top'], state['margin']['right'], state['margin']['bottom'], state['margin']['left']] 
                if all(values):  # All edges are clean
                    stop = True
                elif any(m > 30 for m in margin):
                    stop = True
                else:
                    for k, v in box_new.items():
                        if v:
                            state['margin'][k] = state['margin'][k]
                        else:
                            state['margin'][k] += 2
                print(f"box_new: {box_new}")
                print(f"margin: {state['margin']}")
            # Step 1.6: Remap pixels to cropped canvas
            state["tick_pixel_remap"] = _remap_to_cropped_canvas(state['tick_pixel_raw'], state["chart_border"])
            # Step 1.7: Fit pixel-to-value relationship
            state["pixel_to_value"] = _pixel_tickvalue_fitting(state['tick_pixel_remap'])
            # Step 1.8: Extract curve & convert to grayscale
            curve_points, curve_gray_values = _process_and_extract_curve_points(state['crop_path'])
            state["curve_points"] = curve_points
            state["curve_gray_values"] = curve_gray_values
            # Step 1.9: Reconstruct spectrum
            state["spectrum"] = _convert_to_spectrum(state['curve_points'], state['curve_gray_values'], state['pixel_to_value'])
            # Step 1.10: Detect peaks and troughs
            await self.peak_trough_detection(state)
            print(f"Detected {len(state['merged_peaks'])} peaks and {len(state['merged_troughs'])} troughs.")
            # Step 1.10.5: Continuum fitting
            await self.continuum_fitting(state)
            # Step 1.11: Optional plotting
            if plot:
                try:
                    state["spectrum_fig"] = _plot_spectrum(state)
                except Exception as e:
                    print(f"❌ Plotting spectrum or features failed with error: {e}")
                    raise
            return state
        except Exception as e:
            print(f"Error in spectral visual interpreter: {e}")
            raise

# ---------------------------------------------------------
# 2. Rule-based Analyst — Responsible for rule-based physical analysis
# ---------------------------------------------------------
class SpectralRuleAnalyst(BaseAgent):
    """
    Rule-driven analyst: performs qualitative analysis based on given physical and spectral line knowledge.
    """

    def __init__(self, mcp_manager: MCPManager):
        super().__init__(
            agent_name='Spectral Rule Analyst',
            mcp_manager=mcp_manager
        )

    async def describe_spectrum_picture(self, state: SpectroState):
        function_prompt = state['prompt'][f'{self.agent_name}']['describe_spectrum_picture']
        async def _filter_noise(state):
            band_name = state['band_name']
            band_wavelength = state['band_wavelength']

            if not band_name or not band_wavelength:
                return {
                    "filter_noise": 'false',
                    "filter_noise_wavelength": None
                }
            else:
                # Identify overlapping regions
                overlap_regions = find_overlap_regions(band_name, band_wavelength)
                spec = state['spectrum']
                wl = np.array(spec['new_wavelength'])
                d_f = np.array(spec['delta_flux'])

                system_prompt = function_prompt['_filter_noise']['system_prompt']
                band_name_json = json.dumps(band_name, ensure_ascii=False)
                ham = f"""
The camera/filters used for this spectrum are named:
{band_name_json}
Below is sample data from the spectrum near the boundaries between these camera/filters.
"""
                for key in overlap_regions.keys():
                    overlap = overlap_regions[key]
                    scale = overlap[1] - overlap[0]
                    scale = scale * 2
                    center = (overlap[0] + overlap[1]) / 2
                    left = center - scale / 2
                    right = center + scale / 2
                    mask = (wl >= left) & (wl <= right)
                    wl_t = wl[mask]
                    wl_t = wl_t.tolist()
                    wl_t_json = json.dumps(wl_t, ensure_ascii=False)
                    delta_t = d_f[mask]
                    delta_t = delta_t.tolist()
                    delta_t_json = json.dumps(delta_t, ensure_ascii=False)

                    ham += f"""
Boundary region {key}:
Wavelength: {wl_t_json}
Flux error: {delta_t_json}
"""
                user_prompt = function_prompt['_filter_noise']['user_prompt']
                user_prompt = ham + user_prompt

                response = await self.call_llm_with_context(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    image_path=None,
                    parse_json=True,
                    description="Noise filtering judgment"
                )
                return response

        async def _cleaning(state):
            filter_noise = state['visual_interpretation'][0]
            if not safe_to_bool(filter_noise.get('filter_noise', False)):
                state['cleaned_peaks'] = state['merged_peaks']
                state['cleaned_troughs'] = state['merged_troughs']
            else:
                filter_noise_wl = filter_noise.get('filter_noise_wavelength', [])
                filter_noise_wl = np.array(filter_noise_wl)
                wavelength = np.array(state['spectrum']['new_wavelength'])
                peaks = state['merged_peaks']
                cleaned_peaks = []
                wiped_peaks = []
                for p in peaks:
                    wl = p['wavelength']
                    width = p['width_mean']

                    distance = abs(wl - filter_noise_wl)
                    # If any distance value is less than or equal to the peak width, consider it within a noise region
                    if np.any(distance <= width):
                        is_artifact = True
                    else:
                        is_artifact = False
                    if not is_artifact:
                        if p['width_in_km_s'] is not None and p['wavelength'] > wavelength[0]:
                            if p['width_in_km_s'] > 2000:
                                p['describe'] = 'broad line'
                            elif p['width_in_km_s'] < 1000:
                                p['describe'] = 'narrow line'
                            else:
                                p['describe'] = 'medium-width line'
                            cleaned_peaks.append(p)
                    else:
                        wiped_peaks.append(p)
                state['cleaned_peaks'] = cleaned_peaks
                state['wiped_peaks'] = wiped_peaks

                cleaned_troughs = []
                for t in state['merged_troughs']:
                    wl = t['wavelength']
                    distance = abs(wl - filter_noise_wl)
                    if np.any(distance <= width):
                        is_artifact = True
                    else:
                        is_artifact = False
                    if not is_artifact:
                        if t['width_in_km_s'] is not None and t['wavelength'] > wavelength[0]:
                            if t['width_in_km_s'] > 2000:
                                t['describe'] = 'broad trough'
                            elif t['width_in_km_s'] < 1000:
                                t['describe'] = 'narrow trough'
                            else:
                                t['describe'] = 'medium-width trough'
                        else:
                            t['describe'] = 'unprocessed'
                        cleaned_troughs.append(t)
                state['cleaned_troughs'] = cleaned_troughs
            return state

        async def _visual(state):
            system_prompt = function_prompt['_visual']['system_prompt']

            user_prompt_0 = """
Below is a PNG image of an astronomical spectrum. Based on the image content, qualitatively assess the spectrum’s quality to determine whether it is suitable for subsequent scientific analysis (e.g., redshift measurement, spectral line identification).

Please answer the following questions in order:

1. Signal significance: Are there clear absorption or emission features (i.e., noticeable dips, peaks, or undulations)? Or does the spectrum resemble an almost flat line—suggesting features are "compressed" or "drowned out"?
2. Dynamic range: Is the intensity (y-axis) sufficiently expanded to resolve details? Or is it overly compressed, causing most features to cluster together and become difficult to distinguish, only high noises dominate the spectrum?
"""
            response_0 = await self.call_llm_with_context(
                system_prompt=system_prompt,
                user_prompt=user_prompt_0,
                image_path=state['image_path'],
                parse_json=True,
                description="Visual qualitative description — ori"
            )

            user_prompt_1 = function_prompt['_visual']['user_prompt_continuum']
            response_1 = await self.call_llm_with_context(
                system_prompt=system_prompt,
                user_prompt=user_prompt_1,
                image_path=state['continuum_path'],
                parse_json=True,
                description="Visual qualitative description — continuum"
            )
            
            user_prompt_2 = function_prompt['_visual']['user_prompt_lines']
            response_2 = await self.call_llm_with_context(
                system_prompt=system_prompt,
                user_prompt=user_prompt_2,
                image_path=state['spec_extract_path'],
                parse_json=True,
                description="Visual qualitative description — lines"
            )

            response_0_json = json.dumps(response_0, ensure_ascii=False)
            response_2_json = json.dumps(response_2, ensure_ascii=False)
            user_prompt_3 = f"""
Based on the qualitative descriptions of the spectral signal:

{response_0_json}

{response_2_json}

Determine whether this spectrum is suitable for further quantitative analysis. Please provide a verdict of either "Suitable for use" or "Not suitable for use", along with a brief justification.
"""
            response_3 = await self.call_llm_with_context(
                system_prompt=system_prompt,
                user_prompt=user_prompt_3,
                parse_json=True,
                description="Visual qualitative description"
            )

            user_prompt_4 = function_prompt['_visual']['user_prompt_quality']
            response_4 = await self.call_llm_with_context(
                system_prompt=system_prompt,
                user_prompt=user_prompt_4,
                image_path=state['spec_extract_path'],
                parse_json=True,
                description="Visual qualitative description — quality"
            )
            response_1_json = json.dumps(response_1, ensure_ascii=False)
            response_3_json = json.dumps(response_2, ensure_ascii=False)
            response_4_json = json.dumps(response_4, ensure_ascii=False)
            return '\n'.join([response_0_json, response_1_json, response_2_json, response_3_json, response_4_json])

        async def _integrate(state):
            visual_json = json.dumps(state['visual_interpretation'][1], ensure_ascii=False)

            system_prompt = function_prompt['_integrate']['system_prompt']
            ham = f"""
{visual_json}
"""
            user_prompt_integrate = function_prompt['_integrate']['user_prompt'] + ham
            response = await self.call_llm_with_context(
                system_prompt=system_prompt,
                user_prompt=user_prompt_integrate,
                parse_json=True,
                description="Integrated visual qualitative description"
            )
            return response

        result_filter_noise = await _filter_noise(state)
        state['visual_interpretation'] = [result_filter_noise]
        await _cleaning(state)
        result_visual = await _visual(state)
        state['visual_interpretation'].append(result_visual)
        result_integrate = await _integrate(state)
        state['visual_interpretation'] = result_integrate

        visual_interpretation_path = os.path.join(state['output_dir'], f'{state["image_name"]}_visual_interpretation.txt')
        with open(visual_interpretation_path, 'w', encoding='utf-8') as f:
            json_str = json.dumps(state['visual_interpretation'], indent=2, ensure_ascii=False)
            f.write(json_str)
        print('Finished describe_spectrum_picture')
    
    async def preliminary_classification(self, state: SpectroState) -> str:
        """Preliminary classification: roughly classify the astronomical object based on spectral morphology."""

        continuum_interpretation_json = json.dumps(state['visual_interpretation']['continuum_description'], ensure_ascii=False)
        dataset = os.getenv("DATA_SET", "")
        if dataset == 'CSST':
            # CSST version
            system_prompt = """
You are an experienced astronomical spectral analysis assistant.

Your task is to guess the likely class of the astronomical object based on qualitative descriptions and spectral features.

- If the continuum shows a trend of being higher in the blue end and lower in the red end (i.e., falling), the object is a QSO.
- If the continuum shows a trend of being lower in the blue end, peaking in the middle, and dropping again toward the red end (i.e., rising → falling), the object is also a QSO.
- If the continuum shows a trend of being lower in the blue end and higher in the red end (i.e., rising), the object is a Galaxy.

Compare the likelihoods of these two possibilities and provide your choice.

Output the object type in JSON format as follows:
{
    "type": str  # Possible values: "Galaxy", "QSO"
}

Output only one option. Do not include any other information.
"""
        else:
            # DESI version
            system_prompt = """
You are an experienced astronomical spectral analysis assistant.

Your task is to guess the likely class of the astronomical object based on qualitative descriptions and spectral features.

- If the continuum shows a trend of being higher in the blue end and lower in the red end, the object is a QSO.
- If the continuum shows a trend of being lower in the blue end and higher in the red end, the object is a Galaxy.

Compare the likelihoods of these two possibilities and provide your choice.

Output the object type in JSON format as follows:
{
    "type": str  # Possible values: "Galaxy", "QSO"
}

Output only one option. Do not include any other information.
"""
        user_prompt = f"""
Please analyze the following spectral data:

A previous assistant has already provided a qualitative description of the overall spectral shape:
{continuum_interpretation_json}

Based on this description, guess which class this spectrum likely belongs to.
""" + """
Output in JSON format as follows:
{
    "type": str  // Possible values: "Galaxy", "QSO"
}
"""
        response = await self.call_llm_with_context(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_path=None,
            parse_json=True,
            description="Preliminary classification"
        )
        state['preliminary_classification'] = response

    async def preliminary_classification_monkey(self, state):
        """My dear monkey friend and its typewriter"""
        preliminary_classification_json = json.dumps(state['preliminary_classification'], ensure_ascii=False)
        prompt = f"""
You are an astronomical spectral analysis assistant.
You have received a preliminary guess from another assistant regarding the source type of a spectrum:
{preliminary_classification_json}

Please output the source type indicated in this guess.
""" + """
Format the output as JSON:
{
    "type": str  // Possible values: "Galaxy", "QSO"
}
Do not output anything else.
"""
        response = await self.call_llm_with_context(
            system_prompt='',
            user_prompt=prompt,
            parse_json=True,
            description="Preliminary classification monkey"
        )
        return response

    ###################################
    # QSO part
    ###################################
    async def _QSO(self, state):
        """QSO analysis pipeline"""
        try:
            peaks_info = [
                {
                    "wavelength": pe.get('wavelength'),
                    "flux": pe.get('mean_flux'),
                    "width": pe.get('width_mean'),
                    "width_in_km_s": pe.get('width_in_km_s'),
                    "prominence": pe.get('max_prominence'),
                    "seen_in_max_global_smoothing_scale_sigma": pe.get('max_global_sigma_seen', None),
                    "seen_in_max_local_smoothing_scale_sigma": pe.get('max_roi_sigma_seen', None),
                    "describe": pe.get('describe')
                }
                for pe in state.get('cleaned_peaks', [])[:15]
            ]
            peak_json = json.dumps(peaks_info, ensure_ascii=False)

            # Initialize Lyα candidate list
            Lyalpha_candidate = []
            # 获取光谱波长范围
            wavelengths = state['spectrum']['new_wavelength']
            wl_left = wavelengths[0]
            wl_right = wavelengths[-1]
            mid_wavelength = (wl_left + wl_right) / 2
            dataset = os.getenv("DATA_SET", "")
            is_csst = dataset == 'CSST'
            def check_csst_candidate(peak):
                """检查CSST候选线条件"""
                if peak['width_in_km_s'] is None or peak['width_in_km_s'] < 2000:
                    return False
                # 优先检查全局平滑尺度的信噪比
                if (peak['seen_in_max_global_smoothing_scale_sigma'] is not None and 
                    peak['seen_in_max_global_smoothing_scale_sigma'] > 2):
                    return True
                return False
            def check_desi_candidate(peak):
                """检查DESI候选线条件"""
                if (peak['width_in_km_s'] is None or 
                    peak['width_in_km_s'] < 2000 or 
                    peak['wavelength'] >= mid_wavelength):
                    return False
                # 检查全局平滑尺度的信噪比
                if (peak['seen_in_max_global_smoothing_scale_sigma'] is not None and 
                    peak['seen_in_max_global_smoothing_scale_sigma'] > 2):
                    return True
                return False
            def check_local_snr_candidate(peak):
                """检查局部平滑尺度的信噪比条件（用于备选）"""
                if peak['width_in_km_s'] is None or peak['width_in_km_s'] < 2000:
                    return False
                # 对于DESI，需要额外检查波长条件
                if not is_csst and peak['wavelength'] >= mid_wavelength:
                    return False
                # 检查局部平滑尺度的信噪比
                if (peak['seen_in_max_local_smoothing_scale_sigma'] is not None and 
                    peak['seen_in_max_local_smoothing_scale_sigma'] > 2):
                    return True
                return False

            # 第一轮筛选：使用主条件（全局平滑尺度）
            for peak in peaks_info:
                if is_csst:
                    if check_csst_candidate(peak):
                        Lyalpha_candidate.append(peak['wavelength'])
                else:
                    if check_desi_candidate(peak):
                        Lyalpha_candidate.append(peak['wavelength'])

            # 第二轮筛选：如果第一轮没有找到候选，使用备选条件（局部平滑尺度）
            if not Lyalpha_candidate:
                for peak in peaks_info:
                    if check_local_snr_candidate(peak):
                        Lyalpha_candidate.append(peak['wavelength'])

            state['Lyalpha_candidate'] = Lyalpha_candidate
            # Convert candidate list to JSON and print
            Lyalpha_candidate_json = json.dumps(Lyalpha_candidate, ensure_ascii=False)
            print(f"Lyalpha_candidate: {Lyalpha_candidate}")

            trough_info = [
                {
                    "wavelength": tr.get('wavelength'),
                    "flux": tr.get('mean_flux'),
                    "width": tr.get('width_mean'),
                    "seen_in_scales_of_sigma": tr.get('seen_in_scales_of_sigma')
                }
                for tr in state.get('cleaned_troughs', [])[:15]
            ]
            trough_json = json.dumps(trough_info, ensure_ascii=False)
        except Exception as e:
            logging.error(f"Error in _QSO: {e}")
            raise e

        def _common_prompt_header_QSO(state, include_rule_analysis=True, include_step_1_only=False):
            """Construct common prompt header for each step"""
            try:
                visual_json = json.dumps(state['visual_interpretation'], ensure_ascii=False)
                header = f"""
You are an astronomical spectral analysis assistant.

The following information may come from a QSO spectrum with unknown redshift.

A previous assistant has already provided a preliminary description of this spectrum:
{visual_json}

The wavelength range of this spectrum is from {state['spectrum']['new_wavelength'][0]} Å to {state['spectrum']['new_wavelength'][-1]} Å.
"""

                if include_rule_analysis and state['rule_analysis_QSO']:
                    if include_step_1_only:
                        rule_json = json.dumps(state['rule_analysis_QSO'][0], ensure_ascii=False)
                    else:
                        rule_json = json.dumps("\n".join(str(item) for item in state['rule_analysis_QSO']), ensure_ascii=False)
                    header += f"\nPrevious assistants have already performed some analysis:\n{rule_json}\n"

                tol_pixels = getenv_int("TOL_PIXELS", 10)
                a_x = state['pixel_to_value']['x']['a']
                tol_wavelength = a_x * tol_pixels
                header += f"""
Peak/trough identification was performed using scipy functions on both the original curve and Gaussian-smoothed curves at sigma={state['sigma_list']}.

All discussions about peaks and troughs should be based on the following data:
- Top 10 representative emission lines:
{peak_json}
- Potential absorption lines:
{trough_json}
- Wavelength uncertainty is approximately ±{tol_wavelength/2} Å or larger.
"""
                return header
            except Exception as e:
                logging.error(f"Error in _common_prompt_header_QSO: {e}")
                raise e

        def _common_prompt_tail(step_title, extra_notes=""):
            """Construct common tail for each step, preserving step-specific instructions"""
            try:
                tail = f"""
---

Output format:
{step_title}
...

---

🧭 Notes:
- Non-original computed values should be reported with 3 decimal places.
- Avoid redundant summaries.
- Do not repeat input data line by line.
- Focus on physical reasoning and plausible explanations.
- Ensure the final output is complete and not truncated.
"""
                if extra_notes:
                    tail = extra_notes + "\n" + tail
                return tail
            except Exception as e:
                logging.error(f"Error in _common_prompt_tail: {e}")
                raise e
        
        async def step_1_QSO(state):
            try:
                print("Step 1: Lyα line detection")
                header = _common_prompt_header_QSO(state, include_rule_analysis=False)
                tail = _common_prompt_tail("Step 1: Lyα line detection")
                if len(Lyalpha_candidate) > 0:
                    candidate_str = f"\nAlgorithm-selected Lyα candidates include:\n{Lyalpha_candidate_json}\nYou may also propose other options.\n"
                else:
                    candidate_str = ""

                system_prompt = header + tail
                user_prompt = f"""
Please analyze as follows:

Step 1: Lyα line detection
Assume this spectrum contains a Lyα emission line (λ_rest = 1216 Å):
{candidate_str}
1. Among the prominent, broad peaks visible at large smoothing scales, identify which is most likely the Lyα line.
   - Choose from the provided peak list.
   - When candidate lines have similar widths (within 20 Å), prefer the one with higher flux.
2. Output:
   - Observed wavelength λ_obs
   - Flux
   - Line width
3. Use the tool `calculate_redshift` to compute the redshift z assuming this peak is Lyα.
4. Check whether the blueward (shorter-wavelength) side shows features of the Lyα forest: relatively dense, narrow absorption lines clustered near the blue side of Lyα. Briefly comment if present.
""" 
                
                response = await self.call_llm_with_context(
                    system_prompt=system_prompt, 
                    user_prompt=user_prompt, 
                    parse_json=True, 
                    description="Step 1 Lyα analysis"
                )
                state['rule_analysis_QSO'].append(response)
            except Exception as e:
                logging.error(f"Error in step_1_QSO: {e}")
                raise e

        async def step_2_QSO(state):
            print("Step 2: Analysis of other prominent emission lines")
            try:
                header = _common_prompt_header_QSO(state)
                tail = _common_prompt_tail("Step 2: Analysis of other prominent emission lines")
                system_prompt = header + tail

                band_name = state['band_name']
                band_wavelength = state['band_wavelength']
                if band_name: 
                    overlap_regions = find_overlap_regions(band_name, band_wavelength)
                    # 修复方案1：添加空列表检查
                    wiped_peaks = state.get('wiped_peaks', [])
                    if wiped_peaks:
                        # 只取前5个元素，但确保列表不为空
                        width_means = [wp.get('width_mean') for wp in wiped_peaks[:5] if wp.get('width_mean') is not None]
                        if width_means:
                            wws = np.max(width_means)
                        else:
                            # 处理没有有效width_mean的情况
                            wws = 0  # 或使用默认值，或者抛出更具体的异常
                    else:
                        # 处理wiped_peaks为空的情况
                        wws = 0  # 或使用默认值
                    print(f"wws: {wws}")
                    for key in overlap_regions:
                        range_val = overlap_regions[key]
                        overlap_regions[key] = [range_val[0] - wws, range_val[1] + wws]  # Broaden regions to avoid missing features
                    overlap_regions_json = json.dumps(overlap_regions, ensure_ascii=False)
                    wiped = [
                        {
                            "wavelength": wp.get('wavelength'),
                            "flux": wp.get('mean_flux'),
                            "width": wp.get('width_mean'),
                        }
                        for wp in state.get('wiped_peaks', [])[:5]
                    ]
                    wiped_json = json.dumps(wiped, ensure_ascii=False)
                    advanced = f"""\n    - Note: if any theoretical line falls near the following intervals:\n        {overlap_regions_json}\n    it may have been removed as noise. These removed peaks are:\n        {wiped_json}\n    Please reconsider these when analyzing."""
                else:
                    advanced = ""

                user_prompt = f"""
Please continue the analysis:

Step 2: Analysis of other prominent emission lines
1. Using the redshift from Step 1, use the tool `predict_obs_wavelength` to compute the expected observed positions of the following three major emission lines: C IV 1549, C III] 1909, Mg II 2799.
2. Are there matching peaks in the provided spectrum? {advanced}
3. If matches exist, compute their redshifts using `calculate_redshift`. Output in the format: "Line name -- rest wavelength -- observed wavelength -- redshift".
"""

                response = await self.call_llm_with_context(system_prompt, user_prompt, parse_json=False, description="Step 2 emission line analysis")
                state['rule_analysis_QSO'].append(response)
            except Exception as e:
                logging.error(f"Error in step_2_QSO: {e}")
                raise e

        async def step_3_QSO(state):
            try:
                header = _common_prompt_header_QSO(state)
                tail = _common_prompt_tail("Step 3: Final synthesis")
                system_prompt = header + tail

                user_prompt = """
Please continue the analysis:

Step 3: Final synthesis
1. In Steps 1–2, if:
   - Either C IV or C III] is missing or significantly offset, OR
   - The redshift derived from Lyα is inconsistent with those from other lines,
   then output: “We should prioritize the assumption that the Lyα line was not captured by the peak-finding algorithm,” and terminate Step 3. Do not output anything else.
2. Only if a significant Lyα peak exists AND its redshift is consistent with other lines, proceed as follows:
   - Due to astrophysical phenomena like outflows, adopt the redshift from the **lowest-ionization-state line** among all matched lines as the final redshift. Output this redshift value. (Note: Lyα is less reliable due to asymmetry and broadening.)
"""
                response = await self.call_llm_with_context(system_prompt, user_prompt, parse_json=False, description="Step 3 final synthesis")
                state['rule_analysis_QSO'].append(response)
            except Exception as e:
                logging.error(f"Error in step_3_QSO: {e}")
                raise e
            
        async def step_4_QSO(state):
            try: 
                header = _common_prompt_header_QSO(state, include_step_1_only=True)
                tail = _common_prompt_tail("Step 4: Supplementary analysis (assuming the line selected in Step 1 is NOT Lyα)")
                system_prompt = header + tail

                user_prompt = """
Please continue the analysis:

Step 4: Supplementary analysis (assuming the line selected in Step 1 is NOT Lyα)
- Disregard prior conclusions. Consider that the peak chosen in Step 1 might actually be another major emission line.
  - Assume this peak corresponds to C IV:
      - Output its properties:
          - Observed wavelength λ_obs
          - Flux
          - Line width
          - Use `calculate_redshift` to estimate redshift z based on λ_rest = 1549 Å
      - Use `predict_obs_wavelength` to compute expected positions of other major lines (e.g., Lyα, C III], Mg II) at this redshift. Are matching emission lines present?
      - If Lyα falls within the spectral range, check for its presence.
      - For any plausible matches, compute redshifts using `calculate_redshift` and output in the format: "Line name -- rest wavelength -- observed wavelength -- redshift".
  
  - If this assumption is unreasonable, assume the peak corresponds to C III] or another major line, and repeat the inference. Check for the presence of other expected lines (e.g., Lyα, C III], Mg II) if they fall within the spectral range.

- Note: It is acceptable that some emission lines may be absent due to edge effects or low signal-to-noise ratio.
""" + tail

                response = await self.call_llm_with_context(system_prompt, user_prompt, parse_json=False, description="Step 4 supplementary analysis")
                state['rule_analysis_QSO'].append(response)
            except Exception as e:
                logging.error(f"Error in step_4_QSO: {e}")
                raise e
        
        await step_1_QSO(state)
        await step_2_QSO(state)
        await step_3_QSO(state)
        await step_4_QSO(state)

    async def run(self, state: SpectroState):
        """Execute the full rule-based analysis pipeline"""
        try:
            await self.describe_spectrum_picture(state)
            
            plot_cleaned_features(state)
            
            await self.preliminary_classification(state)
            print(state['preliminary_classification'])

            if state['preliminary_classification']['type'] == "QSO":
                await self._QSO(state)
            return state
        except Exception as e:
            import traceback
            print("❌ An error occurred during spectral analysis:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print("Full traceback:")
            traceback.print_exc()
            raise  # Re-raise so the caller can handle it if needed          

# ---------------------------------------------------------
# 3. Revision Supervisor — Responsible for cross-review and evaluation
# ---------------------------------------------------------
class SpectralAnalysisAuditor(BaseAgent):
    """Review Analyst: Reviews and corrects outputs from other analysis agents"""

    def __init__(self, mcp_manager: MCPManager):
        super().__init__(
            agent_name='Spectral Analysis Auditor',
            mcp_manager=mcp_manager
        )

    def _common_prompt_header(self, state: SpectroState) -> str:
        try:
            peaks_info = [
                {
                    "wavelength": pe.get('wavelength'),
                    "flux": pe.get('mean_flux'),
                    "width": pe.get('width_mean'),
                    "prominence": pe.get('max_prominence'),
                    "seen_in_scales_of_sigma": pe.get('seen_in_scales_of_sigma'),
                    "describe": pe.get('describe')
                }
                for pe in state.get('cleaned_peaks', [])[:15]
            ]
            peak_json = json.dumps(peaks_info, ensure_ascii=False)
            trough_info = [
                {
                    "wavelength": tr.get('wavelength'),
                    "flux": tr.get('mean_flux'),
                    "width": tr.get('width_mean'),
                    "seen_in_scales_of_sigma": tr.get('seen_in_scales_of_sigma'), 
                }
                for tr in state.get('cleaned_troughs', [])[:15]
            ]
            trough_json = json.dumps(trough_info, ensure_ascii=False)
            a = state["pixel_to_value"]["x"]["a"]
            rms = state["pixel_to_value"]["x"]["rms"]
            tolerance = getenv_int("TOL_PIXELS", 10)
            rule_analysis = "\n\n".join(str(item) for item in state['rule_analysis_QSO'])
            prompt_1 = f"""
You are a rigorous [Astronomical Spectral Report Review Analyst].

Objective:
- Review spectral analysis reports or hypotheses produced by other analysts
- Identify logical flaws, computational errors, inconsistencies, or incorrect inferences
- Propose corrections or suggest additional analytical directions

Working Principles:
- Maintain objectivity and critical thinking
- Do not restate the original analysis; only point out issues and provide improvement suggestions
- If the original report is sound, explicitly confirm its validity
- Calculations involving redshift and observed spectral wavelengths must use the tools `calculate_redshift` and `predict_obs_wavelength`. Manual calculations are not allowed.

Output Requirements:
- Use explanatory language
- Clearly list review comments (e.g., "Conclusion premature", "Spectral line interpretation correct")
- Attach improvement suggestions for each identified issue
- Conclude with an overall assessment (reliable / partially credible / unreliable)

Background:
Peak and trough detection was performed using scipy functions on the original spectrum combined with Gaussian-smoothed curves at sigma=2, sigma=4, and sigma=16.
Discussions about peaks and troughs should be based on the following data:
- Representative top 10 emission lines:
{peak_json}
- Potential absorption lines:
{trough_json}

The spectral analysis report provided by other analysts is:

{rule_analysis}

This report retains three decimal places in redshift calculations.

The wavelength range of this spectrum spans from {state['spectrum']['new_wavelength'][0]} Å to {state['spectrum']['new_wavelength'][-1]} Å.
"""
            band_name = state['band_name']
            band_wavelength = state['band_wavelength']
            if band_name: 
                overlap_regions = find_overlap_regions(band_name, band_wavelength)
                # 修复方案1：添加空列表检查
                wiped_peaks = state.get('wiped_peaks', [])
                if wiped_peaks:
                    # 只取前5个元素，但确保列表不为空
                    width_means = [wp.get('width_mean') for wp in wiped_peaks[:5] if wp.get('width_mean') is not None]
                    if width_means:
                        wws = np.max(width_means)
                    else:
                        # 处理没有有效width_mean的情况
                        wws = 0  # 或使用默认值，或者抛出更具体的异常
                else:
                    # 处理wiped_peaks为空的情况
                    wws = 0  # 或使用默认值
                for key in overlap_regions:
                    range = overlap_regions[key]
                    overlap_regions[key] = [range[0]-wws, range[1]+wws] # Broaden the overlap regions to ensure the LLM won't miss them
                overlap_regions_json = json.dumps(overlap_regions, ensure_ascii=False)
                wiped = [
                    {
                        "wavelength": wp.get('wavelength'),
                        "flux": wp.get('mean_flux'),
                        "width": wp.get('width_mean'),
                        # "seen_in_scales_of_sigma": wp.get('seen_in_scales_of_sigma')
                    }
                    for wp in state.get('wiped_peaks', [])[:5]
                ]
                wiped_json = json.dumps(wiped, ensure_ascii=False)
                advanced = f"""If any peaks reported fall near the following intervals:\n    {overlap_regions_json}\nthey may have been mistakenly removed as noise. These removed peaks are:\n      {wiped_json}\nPlease carefully evaluate their potential identification as C IV or C III]."""
            else:
                advanced = ""
            prompt_2 = f"""

I expect the spectral analysis report to align as closely as possible with canonical emission lines such as Lyα, C IV, C III], and Mg II. However, it is acceptable if some lines are missing due to signal truncation at spectral edges or low signal-to-noise ratio (SNR).

Moreover, under poor SNR conditions, line-detection algorithms may be affected, so moderate deviations in line width from expected values are permissible.

If the Lyα line should fall within the spectral range but is not listed in the report, significantly downgrade the report's credibility.

If Lyα is reported, verify its flux relative to other lines (e.g., C IV, C III]). If Lyα flux is markedly lower than those of other lines, note this and reduce the report’s credibility.

Due to astrophysical outflow effects, the redshift derived from the lowest-ionization-state emission line should be adopted as the best estimate for the object's redshift.

Use the tool `QSO_rms` to compute the redshift uncertainty ±Δz:
    - Input parameters:
        wavelength_rest: List[float],  # Rest-frame wavelengths of the lowest-ionization emission lines (Lyα is prone to broadening and should be avoided here; prefer other lines)
        a: float = {a},           
        tolerance: int = {tolerance},     
        rms_lambda = {rms}: float    
"""
            return prompt_1 + advanced + prompt_2
        except Exception as e:
            print(f"Error in _common_prompt_header: {e}")
            return ""

    async def auditing(self, state: SpectroState):
        try:
            system_prompt = self._common_prompt_header(state)

            if state['count'] == 0:
                body = f"""
Please review this analysis report.
"""
            elif state['count']: 
                debate_history_json=''
                for i in range(len(state['auditing_history_QSO'])):
                    auditing_history = state['auditing_history_QSO'][i] 
                    response_history = state['refine_history_QSO'][i]

                    auditing_history_json = json.dumps(auditing_history, ensure_ascii=False)
                    response_history_json = json.dumps(response_history, ensure_ascii=False)

                    debate_history_json += f"Discussion round {i+1}\n you: \n{auditing_history_json}\n\n" + f"Refinement Analyst: \n{response_history_json}\n\n"

                body = f"""
The debate history of you and refinement analyst is listed as follows:
{debate_history_json}

Please respond to the analyst's reply and continue your review.
"""
            user_prompt = body
            response = await self.call_llm_with_context(system_prompt, user_prompt, parse_json=False, description="Report Review")
            state['auditing_history_QSO'].append(response)
        except Exception as e:
            print(f"Error in auditing: {e}")

    async def run(self, state: SpectroState) -> SpectroState:
        if state['preliminary_classification']['type'] == "QSO":
            await self.auditing(state)
        return state

# ---------------------------------------------------------
# 4. Reflective Analyst — Freely responds to reviews and refines analysis
# ---------------------------------------------------------
class SpectralRefinementAssistant(BaseAgent):
    """Refiner: Responds to reviews and improves the analysis"""

    def __init__(self, mcp_manager: MCPManager):
        super().__init__(
            agent_name='Spectral Refinement Assistant',
            mcp_manager=mcp_manager
        )

    def _common_prompt_header(self, state) -> str:
        try:
            peaks_info = [
                {
                    "wavelength": pe.get('wavelength'),
                    "flux": pe.get('mean_flux'),
                    "width": pe.get('width_mean'),
                    "prominence": pe.get('max_prominence'),
                    "seen_in_global_scales_of_sigma": pe.get('max_global_sigma_seen', None),
                    "describe": pe.get('describe')
                }
                for pe in state.get('cleaned_peaks', [])[:15]
            ]
            peak_json = json.dumps(peaks_info, ensure_ascii=False)

            trough_info = [
                {
                    "wavelength": tr.get('wavelength'),
                    "flux": tr.get('mean_flux'),
                    "width": tr.get('width_mean'),
                    "seen_in_scales_of_sigma": tr.get('seen_in_scales_of_sigma')
                }
                for tr in state.get('cleaned_troughs', [])[:15]
            ]
            trough_json = json.dumps(trough_info, ensure_ascii=False)
            rule_analysis = "\n\n".join(str(item) for item in state['rule_analysis_QSO'])
            a = state["pixel_to_value"]["x"]["a"]
            rms = state["pixel_to_value"]["x"]["rms"]
            tolerance = getenv_int("TOL_PIXELS", 10)
            prompt_1 = f"""
You are a reflective [Astronomical Spectral Analyst].

Objective:
- Read and understand another analyst's spectral report
- Read and understand feedback provided by the auditor
- Improve your own or others' prior analysis
- Propose new interpretations or revised conclusions

Working Principles:
- Thoughtfully address each piece of feedback and explicitly state how you have improved the analysis
- If you believe the original conclusion is correct, provide strong justification
- Deliver a more rigorous and comprehensive final analysis
- Calculations involving redshift and observed spectral wavelengths must use the tools `calculate_redshift` and `predict_obs_wavelength`. Manual calculations are not allowed.

Output Requirements:
- Use explanatory language
- List received feedback and your corresponding responses
- Provide a refined summary of the spectral analysis
- Explain what was modified and its scientific rationale

Background:
Peak and trough detection was performed using scipy functions on the original spectrum combined with Gaussian-smoothed curves at sigma=2, sigma=4, and sigma=16.
Discussions about peaks and troughs should be based on the following data:
- Representative top 10 emission lines:
{peak_json}
- Potential absorption lines:
{trough_json}

The spectral analysis report provided by other analysts is:

{rule_analysis}

This report retains three decimal places in redshift calculations.

The wavelength range of this spectrum spans from {state['spectrum']['new_wavelength'][0]} Å to {state['spectrum']['new_wavelength'][-1]} Å.
"""
            band_name = state['band_name']
            band_wavelength = state['band_wavelength']
            if band_name: 
                overlap_regions = find_overlap_regions(band_name, band_wavelength)
                # 修复方案1：添加空列表检查
                wiped_peaks = state.get('wiped_peaks', [])
                if wiped_peaks:
                    # 只取前5个元素，但确保列表不为空
                    width_means = [wp.get('width_mean') for wp in wiped_peaks[:5] if wp.get('width_mean') is not None]
                    if width_means:
                        wws = np.max(width_means)
                    else:
                        # 处理没有有效width_mean的情况
                        wws = 0  # 或使用默认值，或者抛出更具体的异常
                else:
                    # 处理wiped_peaks为空的情况
                    wws = 0  # 或使用默认值
                for key in overlap_regions:
                    range = overlap_regions[key]
                    overlap_regions[key] = [range[0]-wws, range[1]+wws] # Broaden the overlap regions to ensure the LLM won't miss them
                overlap_regions_json = json.dumps(overlap_regions, ensure_ascii=False)
                wiped = [
                    {
                        "wavelength": wp.get('wavelength'),
                        "flux": wp.get('mean_flux'),
                        "width": wp.get('width_mean'),
                        # "seen_in_scales_of_sigma": wp.get('seen_in_scales_of_sigma')
                    }
                    for wp in state.get('wiped_peaks', [])[:5]
                ]
                wiped_json = json.dumps(wiped, ensure_ascii=False)
                advanced = f"""If any peaks reported fall near the following intervals:\n    {overlap_regions_json}\nthey may have been mistakenly removed as noise. These removed peaks are:\n      {wiped_json}\nPlease carefully evaluate their potential identification as C IV or C III]."""
            else:
                advanced = ""

            prompt_2 = f"""

I expect the spectral analysis report to align as closely as possible with canonical emission lines such as Lyα, C IV, C III], and Mg II. However, it is acceptable if some lines are missing due to signal truncation at spectral edges or low signal-to-noise ratio (SNR).

Moreover, under poor SNR conditions, line-detection algorithms may be affected, so moderate deviations in line width from expected values are permissible.

If the Lyα line should fall within the spectral range but is not listed in the report, significantly downgrade the report's credibility.

If Lyα is reported, verify its flux relative to other lines (e.g., C IV, C III]). If Lyα flux is markedly lower than those of other lines, note this and reduce the report’s credibility.

Due to astrophysical outflow effects, the redshift derived from the lowest-ionization-state emission line should be adopted as the best estimate for the object's redshift (Lyα is prone to broadening and should be avoided here; prefer other lines).

Use the tool `QSO_rms` to compute the redshift uncertainty ±Δz:
    - Input parameters:
        wavelength_rest: List[float],  # Rest-frame wavelengths of the lowest-ionization emission lines
        a: float = {a},           
        tolerance: int = {tolerance},     
        rms_lambda = {rms}: float 
"""
            return prompt_1 + advanced + prompt_2
        except Exception as e:
            logging.error(f"Error in _common_prompt_header: {e}")
            raise e

    async def refine(self, state: SpectroState):
        try:
            system_prompt = self._common_prompt_header(state)
            auditing_history = state['auditing_history_QSO'][-1]
            auditing_history_json = json.dumps(auditing_history, ensure_ascii=False)
            if len(state['auditing_history_QSO']) == 1:
                ddd = ''
            elif len(state['auditing_history_QSO']) > 1:
                debate_history_json = ""
                for i in range(len(state['refine_history_QSO'])):
                    auditing_history = state['auditing_history_QSO'][i] 
                    response_history = state['refine_history_QSO'][i]

                    auditing_history_json = json.dumps(auditing_history, ensure_ascii=False)
                    response_history_json = json.dumps(response_history, ensure_ascii=False)

                    debate_history_json += f"Debate round {i+1}: \nAuditor:\n{auditing_history_json}\n\n" + f"You: \n{response_history_json}\n\n"

                    ddd = f"""
Your debate history with the auditor is:
{debate_history_json}

"""

            body = f"""{ddd}
The latest feedback from the reviewing auditor is:
{auditing_history_json}

Please respond to this feedback.
"""
            user_prompt = body
            response = await self.call_llm_with_context(system_prompt, user_prompt, parse_json=False, description="Responding to Review")
            state['refine_history_QSO'].append(response)
        except Exception as e:
            logging.error(f"Error in refine: {e}")
            raise e

    async def run(self, state: SpectroState) -> SpectroState:
        try:
            if state['preliminary_classification']['type'] == "QSO":
                await self.refine(state)
            return state
        except Exception as e:
            import traceback
            print("❌ An error occurred during spectral analysis:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print("Full traceback:")
            traceback.print_exc()
            # Optional: return current state or re-raise the exception
            raise  # Re-raise if you want the caller to also handle the exception

# ---------------------------------------------------------
# 🧩 5. Host Integrator — Synthesizes and summarizes multi-agent perspectives
# ---------------------------------------------------------
class SpectralSynthesisHost(BaseAgent):
    """Synthesis Host: Integrates analyses and conclusions from multiple agents"""

    def __init__(self, mcp_manager: MCPManager):
        super().__init__(
            agent_name='Spectral Synthesis Host',
            mcp_manager=mcp_manager
        )

    def get_system_prompt(self) -> str:
        return f"""
You are a coordinating [Astronomical Spectral Analysis Host].

Objective:
- Consolidate outputs from the visual analyst, rule-based analyst, auditor, and refinement assistant
- Synthesize conclusions from diverse perspectives into a final spectral interpretation
- Clearly highlight points of agreement and disagreement among agents

Working Principles:
- Do not invoke any tools
- Do not blindly follow any single analysis
- Maintain overall scientific rigor and logical consistency
- The final output must be traceable (indicate which agent’s input supports each conclusion)

Output Requirements:
- Use explanatory prose
- Retain three decimal places for numerical values
- Output only the analytical content—do not label the source of each segment
- Provide a final synthesized conclusion with a credibility rating (High / Medium / Low)
- Explicitly state any remaining uncertainties
- Follow the specified output format strictly. Do not include extraneous content.
"""

    async def summary(self, state):
        try:
            preliminary_classification_json = json.dumps(state['preliminary_classification'], ensure_ascii=False)
            visual_interpretation_json = json.dumps(state['visual_interpretation'], ensure_ascii=False)
        except Exception as e:
            print("❌ An error occurred during spectral analysis:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            raise

        prompt_1 = f"""

Visual description of the spectrum:
{visual_interpretation_json}

Preliminary classification:
{preliminary_classification_json}
"""
        system_prompt = self.get_system_prompt() + prompt_1

        if state['preliminary_classification']['type'] == "QSO":
            rule_analysis_QSO = "\n\n".join(str(item) for item in state['rule_analysis_QSO'])
            rule_analysis_QSO_json = json.dumps(rule_analysis_QSO, ensure_ascii=False)
            auditing_QSO = "\n\n".join(str(item) for item in state['auditing_history_QSO'])
            auditing_QSO_json = json.dumps(auditing_QSO, ensure_ascii=False)
            refine_QSO = "\n\n".join(str(item) for item in state['refine_history_QSO'])
            refine_QSO_json = json.dumps(refine_QSO, ensure_ascii=False)
            prompt_2 = f"""

Rule-based analyst's perspective:
{rule_analysis_QSO_json}

Auditor's perspective:
{auditing_QSO_json}

Refinement assistant's perspective:
{refine_QSO_json}
"""
            prompt_3 = f"""

Output format as follows:

- Visual characteristics of the spectrum
- If the the spectrum is suitable for quantitative use
- Analysis report (synthesize all viewpoints from the rule-based analyst, auditor, and refinement assistant; structure output step-by-step)
    - Step 1
    - Step 2
    - Step 3
    - Step 4
- Conclusion
    - Object type (Galaxy or QSO)
    - If the object is a QSO, provide redshift z ± Δz
    - Identified spectral lines (format: Line Name - λ_rest - λ_obs - redshift)
    - Signal-to-noise ratio (SNR) of the spectrum
    - Credibility score (0–4):
        If the spectrum is not suitable for quantitative use, must set:
            - Score 0
        If the spectrum is suitable for quantitative use:
            - Score 3: ≥2 major emission lines identified (e.g., Lyα, C IV, C III], Mg II)
            - Score 2: 1 major line identified with supporting weaker features
            - Score 1: 1 major line identified but no corroborating features
    - Whether human intervention is required:
        - Required if credibility ≤ 2
        - Required if SNR is low and Lyα is absent
        - Otherwise, decide autonomously
"""
            user_prompt = prompt_2 + prompt_3
        else:
            user_prompt = f"""
Output format as follows:

- Visual characteristics of the spectrum
- If the the spectrum is suitable for quantitative use
- Conclusion
    - Object type (Galaxy or QSO)
    - If the object is a QSO, provide redshift z ± Δz
    - Identified spectral lines (format: Line Name - λ_rest - λ_obs - redshift)
    - Signal-to-noise ratio (SNR) of the spectrum
    - Credibility score (0–4):
        - Score 2 if the visual description of the spectrum indicates it is suitable for quantitative use and the spectra is classified as galaxy; otherwise 0
    - Whether human intervention is required:
        - Always required if type is Galaxy
"""
        response = await self.call_llm_with_context(system_prompt, user_prompt, parse_json=False, description="Summary")
        state['summary'] = response

    async def in_brief(self, state):
        summary_json = json.dumps(state['summary'], ensure_ascii=False)
        prompt_type = f"""
You are a coordinating [Astronomical Spectral Analysis Host].

You have already produced a summary for an astronomical spectrum:
{summary_json}

- Please output only the **object type** from the **Conclusion** section (choose one of these three terms: Star, Galaxy, QSO)

- Output format: str
- Do not output any other information
"""
        response_type = await self.call_llm_with_context('', prompt_type, parse_json=False, description="Summary")
        state['in_brief']['type'] = response_type

        prompt_redshift = f"""
You are a coordinating [Astronomical Spectral Analysis Host].

You have already produced a summary for an astronomical spectrum:
{summary_json}

Please extract only the **redshift z** from the **Conclusion** section (do not include ± Δz)

- Output format: float or None
- Do not output any other information
"""
        response_redshift = await self.call_llm_with_context('', prompt_redshift, parse_json=False, description="Summary")
        state['in_brief']['redshift'] = response_redshift

        prompt_rms = f"""
You are a coordinating [Astronomical Spectral Analysis Host].

You have already produced a summary for an astronomical spectrum:
{summary_json}

Please extract only the **redshift uncertainty Δz** from the **Conclusion** section (do not include z)

- Output format: float or None
- Do not output any other information
"""
        response_rms = await self.call_llm_with_context('', prompt_rms, parse_json=False, description="Summary")
        state['in_brief']['rms'] = response_rms

        prompt_human = f"""
You are a coordinating [Astronomical Spectral Analysis Host].

You have already produced a summary for an astronomical spectrum:
{summary_json}

Please indicate whether **human intervention is required**, based on the **Conclusion** section.

- Output only “Yes” or “No”
- Output format: str
- Do not output any other information
"""
        response_human = await self.call_llm_with_context('', prompt_human, parse_json=False, description="Summary")
        state['in_brief']['human'] = response_human

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
            # Optional: return current state or re-raise exception
            raise