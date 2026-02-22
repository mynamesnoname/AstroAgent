Please revise the QSO spectral analysis report according to the reviewers' comments.

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
If any peaks reported fall near the following wavelength intervals:  
{{ overlap_regions | tojson }}  
they may have been mistakenly removed as noise.

These affected peaks are:  
{{ wiped_peaks | tojson }}

Please re-evaluate the possibility that these peaks correspond to C IV or C III].

--------------------------------------
{% endif %}

【Original Spectral Analysis Report】

{{ rule_analysis }}

--------------------------------------

Below is the previous review and response history:

{% if debate_history %}
{% for item in debate_history %}
【Round {{ loop.index }}】

Review Comment:  
{{ item.auditing }}

Response:  
{{ item.response }}

{% endfor %}
{% else %}
(No prior history available)
{% endif %}

--------------------------------------

【Latest Review Comment】

{{ latest_auditing }}

--------------------------------------

Please carry out the following tasks:

1. Respond point-by-point to the latest review comment;
2. Correct any unreasonable line identifications;
3. Re-match spectral lines if necessary;
4. Recalculate the redshift using appropriate tools;
5. Use the `calculate_rms_for_qso_redshift_tool` to compute the redshift uncertainty ±Δz;
6. Provide a revised, complete spectral analysis conclusion;
7. Reassess the overall reliability of the report.

Note:  
The spectral analysis report should aim to match typical emission lines such as Lyα, C IV, C III], and Mg II as closely as possible. However, it is acceptable if some emission lines are missing due to incomplete signal coverage at spectral edges or poor signal-to-noise ratio (SNR).  

Additionally, under low-SNR conditions, line-finding algorithms may be less reliable; thus, moderate deviations in line width from expected values are permissible.  

If the Lyα line is expected to lie within the observed wavelength range but is not reported, significantly downgrade the report’s credibility.  

If Lyα is reported, compare its flux with those of other lines (e.g., C IV, C III]). If the Lyα flux is significantly lower than those of other strong lines, explicitly note this discrepancy and reduce the report’s credibility accordingly.  

Due to astrophysical outflow effects, the redshift derived from the lowest-ionization-state emission line should be adopted as the best estimate of the object's systemic redshift.  

Use the `calculate_rms_for_qso_redshift_tool` to compute the redshift uncertainty ±Δz:  
- Tool inputs:  
    wavelength_rest: List[float]  # Rest-frame wavelengths of the lowest-ionization-state emission lines (avoid Lyα here due to potential broadening effects; prefer other lines when possible)  
    a: float = {{ a }}  
    tolerance: int = {{ tol }}  
    rms_lambda = {{ rms }}: float  

--------------------------------------

## Output Format

List of Responses to Review Comments:
- Issue:
  - Review Comment:
  - Response:
  - Revised: Yes / No
  - Revision Details:

Revised Spectral Analysis Conclusion:

Redshift Result:
- Adopted Emission Line(s):
- z:
- Redshift Uncertainty ±Δz:

Overall Reliability Assessment:
Reliability Rating (Reliable / Partially Reliable / Unreliable):