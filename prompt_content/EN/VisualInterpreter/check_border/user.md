You will receive two images:

1. The original spectral image, which may include plot borders.
2. A matplotlib astronomical spectrum image preprocessed with OCR and OpenCV, where an attempt has been made to crop out the borders and surrounding areas.

Task:  
Determine whether obvious straight-line border remnants remain along the four edges (top, right, bottom, left) of the image  
(e.g., long, straight black or dark line segments, typically part of the outer frame of the coordinate axes).

Judgment criteria:  
- If **no such straight-line segment is visible** along a given edge → "cropped cleanly"  
- If **obvious straight-line segments are still visible** along a given edge → "not cropped cleanly"

Please output your result strictly in the following JSON format, containing only the four specified keys, with values as the strings 'true' or 'false':

{
    "top": "true" or "false",
    "right": "true" or "false",
    "bottom": "true" or "false",
    "left": "true" or "false"
}

Do not output any other content.