The program detected multiple candidate results near the same wavelength that may belong to the same spectral line.

{% if approved_peaks %}
Candidate emission line data is as follows:
{% endif %}

{% if approved_troughs %}
Candidate absorption line data is as follows:
{% endif %}

{{ group_data | tojson }}

Please determine the most likely true spectral line wavelength.

Requirements:
1. Analyze the evidence from different candidates
2. Briefly explain the reasoning for your judgment
3. Select the single most reasonable wavelength

Please return only one JSON object without any additional text.

Output format:

{
  "selected_wavelength": float,
  "selected_index": int,
  "reason": "Brief explanation of why this candidate was selected"
}
