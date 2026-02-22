The qualitative description of the continuum shape of this spectrum is:

{{ continuum_description | tojson }}

{% if dataset == "CSST" %}

- If the continuum shows a higher blue end and a lower red end (overall decreasing trend) → QSO  
- If the continuum shows a lower blue end, a higher middle section, and a decreasing red end (rising then falling) → QSO  
- If the continuum shows a lower blue end and a higher red end (overall increasing trend) → Galaxy  

{% else %}

- If the continuum shows a higher blue end and a lower red end → QSO  
- If the continuum shows a lower blue end and a higher red end → Galaxy  

{% endif %}

Please strictly follow the above rules for classification.