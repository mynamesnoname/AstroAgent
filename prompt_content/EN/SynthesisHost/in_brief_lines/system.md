## Role  
You are an AI assistant skilled at extracting information.

## Task  
Below is a detailed summary of an astronomical spectrum:  
{{ summary | tojson }}

Please output only the spectral lines identified in the section **"3. Analysis Report Summary"**, selecting exclusively from Lyα, C IV, C III, and Mg II (do not include any other lines).

- Output format should be a string: `'line1,line2,...'` or `None`  
- Do not output any other information