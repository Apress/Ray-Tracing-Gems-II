float geom_fact_sa(vec3 P, vec3 P_surf, vec3 n_surf) {
  vec3 dir = normalize(P_surf - P);
  float dist2 = distance_squared(P, P_surf);
  return abs(-dot(n_surf, dir)) / dist2;
}

vec3 direct_light(vec3 P, vec3 n, vec3 wo, Material m) {
  float pdf;
  vec3 l_pos, l_nor, Le;
  if (!sample_lights(P, n, &l_pos, &l_nor, &Le, &pdf)) {
    return vec3(0.0);
  }
  float G = geom_fact_sa(P, l_pos, l_nor);
  vec3 wi = normalize(l_pos - P);
  vec3 brdf = evaluate_material(m, n, wo, wi);
  return brdf*G*clamped_dot(n, wi)*Le/pdf;
}
