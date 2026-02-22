The camera/filters name for this spectrum is:  
{{ arm_name | tojson }}

Below are sample data from the spectral overlap regions between arms:

{% for region in overlap_regions %}
Overlap region {{ region.region }}:  
Wavelength: {{ region.wavelength | tojson }}  
Flux: {{ region.flux | tojson }}  
Flux error: {{ region.delta_flux | tojson }}

{% endfor %}

Please assess:  
Is there a noticeable, non-physical increase in noise in the overlap regions?