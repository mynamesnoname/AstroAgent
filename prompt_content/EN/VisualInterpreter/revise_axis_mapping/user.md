Please check the following scale value to pixel mapping:  
{{ axis_mapping | tojson(indent=2) }}

Tasks:  
- Revise any entries that violate the monotonicity rule  
- Keep null values unchanged  
- Output the revised JSON array; if the original input is correct, return it as-is  
- Do not output any explanations or additional text