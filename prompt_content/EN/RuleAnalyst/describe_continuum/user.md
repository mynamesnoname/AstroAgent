This is a continuum image extracted from an astronomical spectrum.

## Continuum Trend Analysis

Please use purely qualitative language to determine:

1. The relative flux level ranking (high / medium / low) for the blue end, middle section, and red end.
2. The overall trend from the blue end to the middle section (increasing / decreasing).
3. The overall trend from the middle section to the red end (increasing / decreasing).

Strictly output the result in the following JSON format:

{
  "blue_end": "high" | "medium" | "low",
  "blue_to_mid_trend": "increasing" | "decreasing",
  "mid_section": "high" | "medium" | "low",
  "mid_to_red_trend": "increasing" | "decreasing",
  "red_end": "high" | "medium" | "low"
}

Do not output any other content.