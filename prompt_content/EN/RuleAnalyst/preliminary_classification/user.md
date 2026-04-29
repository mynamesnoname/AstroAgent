Qualitative description of the spectrum continuum:

{{ continuum_description | tojson }}

---

Qualitative description of the spectrum's emission/absorption features:

{{ feature_description | tojson }}

---

Specific information on the spectrum's emission/absorption features:

- Peaks
{% if peaks %}
{% for p in peaks %}

Wavelength: {{ p.wavelength }}
Amplitude: {{ p.amplitude }}
Amplitude rank: {{ p.amplitude_rank }}
Flux at center: {{ p.flux_at_center }}
Width in Å: {{ p.FWHM_A }}
Width in km/s: {{ p.FWHM_km_s }}
Width class: {{ p.width_class }}
Covered troughs (Is there any trough covered by the peak?): {{ p.covered_troughs }}
Covered trough centers: {{ p.trough_centers }}
Neighbors:
{{ p.left_neighbor }}
{{ p.right_neighbor }}
Does it touch the edge: {{ p.quality_boundary_touch }}
------------------------------------------------------
{% endfor %}
{% else %}
No significant peak features
{% endif %}
------------------------------------------------------
------------------------------------------------------
- Troughs
{% if troughs %}
{% for t in troughs %}

Wavelength: {{ t.wavelength }}
Amplitude: {{ t.amplitude }}
Amplitude rank: {{ t.amplitude_rank }}
Flux at center: {{ t.flux_at_center }}
Width in Å: {{ t.FWHM_A }}
Width in km/s: {{ t.FWHM_km_s }}
Neighbors:
{{ t.left_neighbor }}
{{ t.right_neighbor }}
-------------------------------------------------------
{% endfor %}
{% else %}
No significant trough features
{% endif %}

Please qualitatively classify the spectrum based on the above information and provide the reasoning for the classification.
