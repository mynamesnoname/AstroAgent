This is an emission/absorption feature extracted from an astronomical spectrum.

## Emission/Absorption Features
The following are some notable peaks or troughs:
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

Please output your summary based on the data, which must include the following:
- How many broad peaks, intermediate peaks, and narrow peaks does this image contain?
- Which are the notable peaks or troughs?
- Which peaks or troughs might be spurious? (e.g., a peak covering two or more troughs, continuum between two troughs misidentified as a peak, or peaks ranked low with amplitudes very small compared to the highest peak/trough, etc.)
- Could there be doublets?
    - Given that the data are in the observed frame and the redshift is unknown, the doublet separation will definitely be larger than the rest-frame typical separation. It can be considered that the doublet separation ranges from a few tens to over a hundred Å, but generally not exceeding 200 Å.
    - If in a potential doublet system there is a peak with significantly lower amplitude, you must note it.
    - Please use the tool `calculate_peak_amplitude_ratio` to compute the amplitude ratio of the potential doublet system.
- If there are 3 or more absorption lines, how are they distributed in wavelength? Are they obviously dense or relatively uniform?
- Are there any other noteworthy features?

- If no line information exists, output "No spectral lines".

Example:
This spectrum contains AAA broad peaks, BBB intermediate peaks, and CCC narrow peaks. The most notable peaks are at DDD Å (amplitude rank 1), EEE Å (rank 2), and FFF Å (rank 3). Low-ranked peaks (such as ZZZ Å, YYY Å, XXX Å, all with low amplitudes) may be spurious. The separation between GGG Å and HHH Å is about III Å, well beyond a plausible doublet separation, ruling out a doublet; but JJJ Å and KKK Å (low amplitude) have a separation of LLL, which is reasonable, however the KKK line has a lower amplitude. Their amplitude ratio is Amplitude_JJJ / Amplitude_KKK = MMM (please note here the wavelength order of the compared lines). The separation between NNN Å and OOO Å is PPP Å, with an amplitude ratio of Amplitude_NNN / Amplitude_OOO = QQQ (also note here the wavelength order). No absorption lines. It is noteworthy that all peaks are emission features and no trough-covering phenomenon.

Do not output any other content.