#include "functions.h"

float sample_material_pdf(Material m, vec3 normal, vec3 wo, vec3 wi) {
  float d=dot(wi, normal);
	if (d<0.0) return 0.0;
	return d/PI;
}

void sample_material(Material m, vec3 normal, vec3 wo, vec3 *wi, float *out_pdf) {
	vec3 dir = random_cosine_hemisphere(normal);
	*out_pdf = dot(dir, normal)/PI;
	*wi = dir;
}

vec3 evaluate_material(Material m, vec3 normal, vec3 wo, vec3 wi) {
	float d=dot(wi, normal);
	if (d<0.0) return vec3(0.0);
	return m.diffuse/PI;
}
