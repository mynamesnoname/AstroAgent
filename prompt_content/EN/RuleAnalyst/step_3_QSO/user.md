Step 3 Task: Comprehensive Assessment of Redshift Consistency

--------------------------------------
Overall Spectral Description:
{{ qualitative_analysis | tojson }}

Wavelength Range:
{{ wl_left }} Å – {{ wl_right }} Å

Gaussian Smoothing Scale(s):
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

1. During Steps 1 to 2, if either of the following conditions occurs:
    - The two primary emission lines, C IV and C III], are missing or significantly offset,
    - The redshift derived from the Lyα line is inconsistent with those derived from other lines,  
then output **"Lyα should be assumed as not captured by the peak-finding algorithm"** and terminate the Step 3 analysis immediately. Do not provide any additional information.

2. Only proceed with the following step if a significant Lyα emission peak is present **and** its derived redshift is generally consistent with those from other lines:
    - Due to astrophysical phenomena such as outflows, adopt the redshift corresponding to the **lowest-ionization-state emission line** among all current matches as the final spectral redshift. Output this redshift value. (Note: Lyα has lower reliability due to potential asymmetry and broadening.)