Step 2 Task: Verification of Matches for Other Prominent Emission Lines

--------------------------------------
Overall Spectral Description:
{{ qualitative_analysis | tojson }}

Wavelength Range:
{{ wl_left }} Å – {{ wl_right }} Å

Gaussian Smoothing Scale:
{{ sigma_list }}

Representative Emission Peaks:
{{ peak_list | tojson }}

Absorption Troughs:
{{ trough_list | tojson }}

Relative Measurement Uncertainty in Peak Wavelengths:
± {{ tol_wavelength }} Å

History:

{{ history | tojson }}

--------------------------------------

Target Emission Lines:

- C IV (1549 Å)
- C III] (1909 Å)
- Mg II (2799 Å)

Task:

Step 2: Analysis of Other Significant Emission Lines  
1. Using the redshift obtained in Step 1, calculate the theoretical observed wavelengths of the following three primary emission lines—C IV 1549, C III] 1909, and Mg II 2799—in the spectrum using the tool `predict_obs_wavelength`.  
2. Are there peaks in the provided spectrum that match these three theoretical positions?  
{% if overlap_regions %}
【Important Note: Potentially Removed Peaks】

The following peaks may have been removed as noise:

{{ wiped_peaks | tojson }}

If any theoretical wavelength falls within these intervals,  
please take this into account before deciding on a match.
{% endif %}
3. If matches exist between emission lines and observed peaks, compute the redshift for each match using the tool `calculate_redshift`. Report results in the format:  
"Emission Line Name -- Rest Wavelength -- Observed Wavelength -- Redshift".

If no matches are found, explicitly state so.