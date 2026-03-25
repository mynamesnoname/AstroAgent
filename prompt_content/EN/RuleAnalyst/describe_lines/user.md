This is a continuum image extracted from an astronomical spectrum.

## Emission/Absorption Features
Below are some prominent peaks or troughs:
- Peaks
{% if cleaned_peaks %}
{% for p in cleaned_peaks %}
Wavelength: {{ p.wavelength }}
Flux:  {{ p.flux }}
Prominence: {{ p.prominence }}
Width: {{ p.describe }}

{% endfor %}
{% else %}
No prominent peaks found.
{% endif %}

- Troughs
{% if cleaned_troughs %}
{% for t in cleaned_troughs %}
Wavelength: {{ t.wavelength }}
Flux:  {{ t.flux }}
Depth: {{ t.depth }}
Width: {{ t.describe }}

{% endfor %}
{% else %}
No prominent troughs found.
{% endif %}

Please summarize based on the data and image:
- Which dominates: broad lines (>2000 km/s) or narrow lines (<1000 km/s)?
- Are there any asymmetric peaks?

Strictly output the result in the following JSON format:

{
    "dominant_line_type": "broad/narrow",
    "asymmetric_peaks_wavelength": List[float] or null
}

Do not output any other content.