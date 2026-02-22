Please conduct a rigorous review of the following QSO spectral analysis report.

--------------------------------------
【Basic Spectral Data】

Wavelength range:  
{{ wl_left }} Å – {{ wl_right }} Å

Representative emission lines:  
{{ peak_list | tojson }}

Possible absorption features:  
{{ trough_list | tojson }}

--------------------------------------

{% if overlap_regions %}
If any reported peaks fall near the following wavelength intervals:  
{{ overlap_regions | tojson }}  
they may have been mistakenly removed as noise.

These affected peaks are:  
{{ wiped_peaks | tojson }}

Please carefully evaluate whether these peaks could correspond to C IV or C III].

--------------------------------------

{% endif %}
【Reports from Other Analysts】

{{ rule_analysis }}

--------------------------------------

Review and response history from prior rounds:

{% if debate_history %}
{% for item in debate_history %}
【Round {{ loop.index }}】

Review comments:  
{{ item.auditing }}

Response:  
{{ item.response }}

{% endfor %}
{% else %}
(No prior history available)
{% endif %}

--------------------------------------

Please perform the following checks:

1. Assess whether the identified spectral lines are reasonable;  
2. Verify consistency in the redshift calculation;  
3. Check for missing key emission lines (Lyα, C IV, C III], Mg II);  
4. Evaluate whether the relative flux ratios among emission lines are physically plausible;  
5. Recalculate the redshift using appropriate tools if necessary;  
6. Use the `calculate_rms_for_qso_redshift_tool` to estimate the redshift uncertainty.

Note:  
The spectral analysis report should aim to match typical emission lines such as Lyα, C IV, C III], and Mg II as closely as possible. However, it is acceptable for some lines to be undetected due to incomplete spectral coverage at the edges or low signal-to-noise ratio (SNR).  

Under low-SNR conditions, line-detection algorithms may also be less reliable; thus, moderate deviations in line width from expected values are permissible.  

If the Lyα line is expected to lie within the observed wavelength range but is not reported, significantly downgrade the report’s credibility.  

If Lyα is reported, compare its flux with those of other lines (e.g., C IV, C III]). If the Lyα flux is substantially lower than those of higher-ionization lines, explicitly note this anomaly and reduce the report’s credibility accordingly.  

Due to astrophysical outflow effects, the redshift derived from the lowest-ionization emission line should be adopted as the best estimate of the systemic redshift.  

Use the `calculate_rms_for_qso_redshift_tool` to compute the redshift uncertainty ±Δz, with the following inputs:  
- `wavelength_rest`: List[float] — rest-frame wavelengths of the lowest-ionization emission lines (avoid Lyα here due to its susceptibility to broadening; prefer other lines when possible)  
- `a`: float = {{ a }}  
- `tolerance`: int = {{ tol }}  
- `rms_lambda`: float = {{ rms }}

--------------------------------------

## Output Format

List of Issues:  
- Issue:  
  - Description:  
  - Impact:  
  - Recommendation:  

Validation of Reasonableness (if applicable):

Overall Assessment:  
Credibility Rating (Reliable / Partially Reliable / Unreliable):