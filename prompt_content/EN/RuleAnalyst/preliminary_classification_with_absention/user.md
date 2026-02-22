A qualitative description of the spectral continuum is as follows:

{{ continuum_description | tojson }}

--------------------------------------------------
Continuum classification rules:

{% if dataset == "CSST" %}
- Higher at the blue end and lower at the red end (decreasing trend) → QSO
- Lower at the blue end, higher in the middle, and decreasing toward the red end (rising then falling) → QSO
- Lower at the blue end and higher at the red end (increasing trend) → Galaxy
{% else %}
- Higher at the blue end and lower at the red end → QSO
- Lower at the blue end and higher at the red end → Galaxy
{% endif %}

--------------------------------------------------
Signal-to-noise ratio (SNR) rules:

The maximum SNR of this spectrum is {{ snr_max }}.

{% if snr_threshold_upper %}
- If the maximum SNR > {{ snr_threshold_upper }}, the output must be either "QSO" or "Galaxy".
- If {{ snr_threshold_lower }} < maximum SNR ≤ {{ snr_threshold_upper }},
  you may choose from "QSO", "Galaxy", or "Unknown".
  - The closer the SNR is to {{ snr_threshold_upper }}, the more confident the classification should be.
  - The lower the SNR, the more likely "Unknown" should be selected.
- If the maximum SNR ≤ {{ snr_threshold_lower }}, the output must be "Unknown".
{% else %}
(No SNR thresholds provided; classification based solely on the continuum.)
{% endif %}

--------------------------------------------------

Please output your judgment process and spectral category.
