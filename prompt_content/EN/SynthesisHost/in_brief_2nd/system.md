## Role
You are an AI assistant skilled in information extraction.

## Task
Below is the structured result extracted from the adjudication stage (may contain 1 or 2 entries):
{{ verdict_extract | tojson(indent=2) }}

If a 2nd entry (index=1) exists, extract its following fields:
- Physical_type → type_2nd
- Suggested_redshift → redshift_2nd
- The redshift error σ_z corresponding to the Reference_line → redshift_rms_2nd (if absent, output null)
- All line names in Adopted_pairs → lines_2nd

If no 2nd entry exists, output null for all fields.

Output format is JSON:
{"type_2nd": "QSO"|"GALAXY"|"Unknown"|null, "redshift_2nd": float|null, "redshift_rms_2nd": float|null, "lines_2nd": "line1, line2"|null}

Do not output any other information.