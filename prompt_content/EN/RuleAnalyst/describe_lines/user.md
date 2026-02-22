This is a continuum image extracted from an astronomical spectrum.

## Emission/Absorption Features
Below are some prominent peaks or troughs:
- Peaks
{% for p in peaks %}
Wavelength: {{ p.wavelength }}
Flux:  {{ p.mean_flux }}
Prominence: {{ p.max_prominence }}
Width: {{ p.describe }}

{% endfor %}

- Troughs
{% for t in troughs %}
Wavelength: {{ t.wavelength }}
Flux:  {{ t.mean_flux }}
Depth: {{ t.max_depth }}
Width: {{ t.describe }}

{% endfor %}

Please summarize based on the data and image:
- Which dominates: broad lines (>2000 km/s) or narrow lines (<1000 km/s)?
- Are there any asymmetric peaks?

Strictly output the result in the following JSON format:

{
    "dominant_line_type": "broad/narrow",
    "asymmetric_peaks_wavelength": List[float] or null
}

Do not output any other content.