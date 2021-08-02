vec3 direct_mat(vec3 P, vec3 n, vec3 wo, Material m) {
  vec3 wi;
  float pdf;
  if (!sample_material(m, n, wo, &wi, &pdf)) {
    return vec3(0.0);
  }
  Intersect i = intersect(P, wi);
  if (!i.hit) return vec3(0.0);
  vec3 brdf = evaluate_material(m, n, wo, wi);
  vec3 Le = evaluate_emissive(i, wi);
  return brdf*dot(wi, n)*Le/pdf;
}