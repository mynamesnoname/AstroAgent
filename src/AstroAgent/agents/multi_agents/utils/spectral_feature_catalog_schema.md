# Spectral Feature Catalog Schema

This document defines the output structure for spectral line detection (emission peaks and absorption troughs). The catalog is designed to be LLM-friendly, providing clear physical context and human-readable quality indicators for downstream reasoning.

---

## Overview

Each detected spectral feature (peak or trough) is represented as a structured record containing the following sections:

1. **Basic Identity** – Position in the spectrum and detection iteration
2. **Position & Morphology** – Wavelength and line width
3. **Intensity** – Amplitude, flux, and integrated quantities
4. **Significance** – Statistical measures of detection confidence
5. **Quality Control** – Human-readable boolean flags for data quality
6. **Classification Summary** – Interpreted properties (width class, pseudo-peak status)

---

## Field Definitions

### 1. Basic Identity

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `index` | int | – | Index of the wavelength in the global spectrum array |
| `iteration` | int | – | Detection iteration round (1, 2, 3, ...) |

---

### 2. Position & Morphology

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `wavelength` | float | Å | Central wavelength of the fitted Gaussian |
| `wavelength_err` | float | Å | Uncertainty in the fitted central wavelength (±) |
| `FWHM_A` | float | Å | Full Width at Half Maximum in angstroms |
| `FWHM_km_s` | float | km/s | Full Width at Half Maximum in velocity units. Calculated as: `FWHM_km_s = FWHM_A / wavelength × 299792.458` |

**Width Classification Thresholds:**
- `narrow` : FWHM < 1000 km/s
- `intermediate` : 1000 ≤ FWHM < 2000 km/s  
- `broad` : FWHM ≥ 2000 km/s

---

### 3. Intensity

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `amplitude` | float | – | **Emission lines**: Peak height above continuum (positive value). **Absorption lines**: Trough depth below continuum (positive value, representing absorption depth). |
| `amplitude_err` | float | – | Uncertainty in the fitted amplitude/depth (±) |
| `integrated_flux` | float | Jy·Å | **Emission lines**: Integrated flux under the Gaussian curve. **Absorption lines**: Equivalent width (EW), representing the integrated absorption area. |
| `flux_at_center` | float | Jy | Original spectrum flux value at the central wavelength (before continuum subtraction) |

---

### 4. Significance

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `global_delta_chi2` | float | – | Δχ² from global refinement fit. Measures how much the Gaussian model improves the fit across the full spectrum. Higher values indicate more significant detections. Normalized by number of data points. |
| `local_delta_chi2` | float | – | Δχ² from local window fit. Measures fit improvement within the detection window. Useful for comparing detections from different window sizes. |

**Interpretation Guide:**
- Δχ² > 0.3 : High significance
- 0.1 < Δχ² ≤ 0.3 : Moderate significance
- Δχ² ≤ 0.1 : Low significance (may be noise)

---

### 5. Quality Control

A dictionary of boolean flags for data quality assessment. Each flag is self-explanatory and requires no additional interpretation.

| Field | Type | Description |
|-------|------|-------------|
| `low_delta_chi2` | bool | `True` if Δχ² is below the significance threshold, indicating a potentially spurious detection |
| `boundary_touch` | bool | `True` if fitted parameters (FWHM or amplitude) hit the allowed bounds, suggesting possible fitting issues |
| `large_error` | bool | `True` if parameter uncertainties are large relative to the parameter values, indicating unreliable fit |
| `blended` | bool | `True` if this feature overlaps with another detected feature within the blend tolerance distance |
| `low_snr_depth` | bool | **Absorption lines only**: `True` if the depth signal-to-noise ratio is below threshold, suggesting the trough may be noise |

**Example:**
```json
{
  "quality": {
    "low_delta_chi2": false,
    "boundary_touch": true,
    "large_error": false,
    "blended": false,
    "low_snr_depth": false
  }
}
```

---

### 6. Classification Summary

| Field | Type | Description |
|-------|------|-------------|
| `width_class` | string | Width classification: `"narrow"`, `"intermediate"`, or `"broad"` based on FWHM_km_s thresholds |
| `is_pseudo_peak` | bool | **Emission lines only**: `True` if this peak is flagged as a potential pseudo-peak artifact |
| `pseudo_reason` | string or null | **Emission lines only**: Explanation for pseudo-peak classification. `null` if not a pseudo-peak. Examples: `"covers 2 troughs at [4500.0, 4520.0] Å"`, `"low delta_chi2"`, `"boundary touch"` |

**Pseudo-Peer Detection Logic:**
A peak is flagged as a pseudo-peak candidate if its ±3σ range covers 2 or more absorption troughs. This often indicates a "ridge" between valleys rather than a true emission feature.

---

## Complete Example Record

### Emission Line (Peak)

```json
{
  "index": 4523,
  "iteration": 1,
  
  "wavelength": 5500.0,
  "wavelength_err": 0.5,
  "FWHM_A": 15.0,
  "FWHM_km_s": 817.0,
  
  "amplitude": 2.5,
  "amplitude_err": 0.1,
  "integrated_flux": 42.3,
  "flux_at_center": 3.2,
  
  "global_delta_chi2": 0.45,
  "local_delta_chi2": 0.38,
  
  "quality": {
    "low_delta_chi2": false,
    "boundary_touch": false,
    "large_error": false,
    "blended": false,
    "low_snr_depth": false
  },
  
  "summary": {
    "width_class": "narrow",
    "is_pseudo_peak": false,
    "pseudo_reason": null
  }
}
```

### Absorption Line (Trough)

```json
{
  "index": 3891,
  "iteration": 2,
  
  "wavelength": 4200.0,
  "wavelength_err": 0.3,
  "FWHM_A": 8.5,
  "FWHM_km_s": 606.0,
  
  "amplitude": 1.8,
  "amplitude_err": 0.2,
  "integrated_flux": 17.2,
  "flux_at_center": 1.5,
  
  "global_delta_chi2": 0.52,
  "local_delta_chi2": 0.41,
  
  "quality": {
    "low_delta_chi2": false,
    "boundary_touch": false,
    "large_error": false,
    "blended": true,
    "low_snr_depth": false
  },
  
  "summary": {
    "width_class": "narrow"
  }
}
```

---

## Notes for LLM Reasoning

1. **Amplitude Sign Convention**: Both emission peaks and absorption troughs use positive `amplitude` values. For emission, this is peak height above continuum; for absorption, this is trough depth below continuum.

2. **Quality Flags**: Use these to filter or weight features in downstream analysis. Features with `low_delta_chi2`, `large_error`, or `low_snr_depth` should be treated with caution.

3. **Pseudo-Peaks**: When `is_pseudo_peak` is `True`, the feature likely represents a "ridge" between absorption troughs rather than a real emission line. Check `pseudo_reason` for the specific cause.

4. **Width Class**: The `width_class` field provides a quick categorization useful for identifying line types (e.g., narrow lines often indicate star formation, while broad lines suggest AGN activity).

5. **Iteration**: Features detected in later iterations are typically weaker or were masked by stronger features in earlier rounds.

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| v1.0 | 2026-04-13 | Initial schema definition |
