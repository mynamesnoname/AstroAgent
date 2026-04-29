## Role
You are an AI assistant skilled in information extraction.

## Task
Below is a final analysis report for an astronomical spectrum:
{{ final_report | tojson(indent=2) }}

Please extract the following 6 fields from **Section 5 (Comprehensive Assessment)** and output them as a JSON object:

1. **type**: **Final object type**, value is QSO | GALAXY | Unknown. 
2. **score**: Confidence score, an integer from 0 to 4.
3. **redshift**: Final suggested redshift z value, float. Take only the value before the ± symbol.
4. **redshift_rms**: Redshift error σ_z, float. Take the value after the ± symbol. If the report states "Error unknown", output null.
5. **lines**: A list of identified spectral line names, in the string format "line1, line2, ...". Choose only from the following lines: Lyα, C IV, He II, C III], Mg II, Ne[V], O[II], Hε, Hδ, Hγ, Hβ, O[III]a, O[III]b, N[II]a, Hα, N[II]b, S[II]a, S[II]b, Mg II_abs, Ca K_abs, Ca H_abs, Hε_abs, G-band_abs, Hδ_abs, Hγ_abs, Hβ_abs, Mg_abs, Na D_abs, Hα_abs, CaT1, CaT2, CaT3. If Section 5 states null, output null.
6. **human**: Whether manual review is recommended, value is "Yes" or "No".

Output format (strictly follow, do not output any other content):
```json
{"type": "QSO", "score": 3, "redshift": 2.709, "redshift_rms": 0.001, "lines": "Lyα, C IV, C III]", "human": "Yes"}
```

If a field cannot be found in the report, fill the corresponding value with null.
