Spectral information is as follows:

Wavelength range:
{{ wl_left }} Å – {{ wl_right }} Å

Peak wavelength relative measurement error:
± {{ tol_wavelength }} Å

---

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

---

Preliminary classification and feature description of the spectrum:

{{ preliminary_classification | tojson }}

---

Current brute-force line-matching hypothesis to be analyzed (No. {{ hypothesis_index }} of {{ hypothesis_total }}):

Hypothesis: {{ match.Hypothesis }}
z_max: {{ match.z_max }}  z_min: {{ match.z_min }}  z_spread: {{ match.z_spread }}
N_emission: {{ match.N_emission }}  N_absorption: {{ match.N_absorption }}
Redshift warning: {{ match["Redshift warning"] }}
Emission matches:
{% for em in match["Emission matches"] %}
  {{ em }}
{% endfor %}
Absorption matches:
{% for ab in match["Absorption matches"] %}
  {{ ab }}
{% endfor %}
Missing emission lines (in obs range but not matched):
{% if match["Missing emission lines"] %}
{% for ml in match["Missing emission lines"] %}
  {{ ml }}
{% endfor %}
{% else %}
  (none)
{% endif %}

---

Please complete the analysis of the current hypothesis in the order of Step F-1 to Step F-3.