vec3 direct_cos(vec3 P, vec3 n, vec3 wo, Material m) {
  vec3 wi = random_cosine_hemisphere(n);
  float pdf = dot(wi, n)/PI;
  Intersect i = intersect(P, wi);
  if (!i.hit) return vec3(0.0);
  vec3 brdf = evaluate_material(m, n, wo, wi);
  vec3 Le = evaluate_emissive(i, wi);
  return brdf*dot(wi, n)*Le/pdf;
}