## Role  
You are an AI assistant skilled at extracting information.

## Task  
Below is a detailed summary of an astronomical spectrum:  
{{ summary | tojson }}

Please output only the **redshift error Δz** from the section **"3. Analysis Report Summary"** (do not output z).

- Output format should be a float or None  
- Do not output any other information