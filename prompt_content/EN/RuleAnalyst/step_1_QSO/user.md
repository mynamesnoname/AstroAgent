Step 1 Task: Lyα Emission Line Detection and Initial Redshift Estimation

--------------------------------------

Overall spectral description:  
{{ qualitative_analysis | tojson }}

Wavelength range:  
{{ wl_left }} Å – {{ wl_right }} Å

Gaussian smoothing scales:  
{{ sigma_list }}

Representative emission peaks:  
{{ peak_list | tojson }}

Absorption troughs:  
{{ trough_list | tojson }}

Relative measurement uncertainty in peak wavelength:  
± {{ tol_wavelength }} Å

--------------------------------------

Assume the presence of a Lyα emission line (λ_rest = 1216 Å) in the spectrum.

{% if lyalpha_candidates %}
Candidate Lyα lines:  
{{ lyalpha_candidates | tojson }}
{% endif %}

1. Among the prominent, broad peaks visible at large smoothing scales with relatively high flux, identify the most likely Lyα emission line:
   - Select from the provided list of peaks.
   - When candidate lines have similar widths (within 20 Å), prefer the peak with higher flux.

2. Output the following for the selected peak:
   - Observed wavelength (λ_obs)
   - Flux
   - Line width

3. Use the `calculate_redshift` tool to compute the redshift *z* assuming this peak corresponds to the Lyα emission line.

4. Examine the blue side (shorter-wavelength side) of the candidate Lyα line for signatures of the Lyα forest: look for a region with relatively dense, narrow absorption features clustered near the blue side of the Lyα line. Briefly describe any such features you observe.