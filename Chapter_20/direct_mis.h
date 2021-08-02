vec3 direct_mis(vec3 P, vec3 n, vec3 wo, Material m) {
  vec3 result = vec3(0.0);
  vec3 Le, m_wi, l_pos, l_nor;
  float l_pdf, m_pdf;
  // Light sampling
  if (sample_lights(P, n, &l_pos, &l_nor, &Le, &l_pdf)) {
    vec3 l_wi = normalize(l_pos - P);
    float G=geom_fact_sa(P, l_pos, l_nor);
    float m_pdf=sample_material_pdf(m, n, wo, l_wi);
    float mis_weight=balance_heuristic(l_pdf, m_pdf*G);
    vec3 brdf=evaluate_material(m, n, wo, l_wi);
    result+=brdf*mis_weight*G*clamped_dot(n, l_wi)*Le/l_pdf;
  }
  // Material sampling
  if (sample_material(m, n, wo, &m_wi, &m_pdf)) {
    Intersect i = intersect(P, m_wi);
    if (i.hit && i.mat.is_emissive) {
      float G=geom_fact_sa(P, i.pos, i.nor);
      float light_pdf=sample_lights_pdf(P, n, i);
      float mis_weight=balance_heuristic(m_pdf*G, light_pdf);
      vec3 brdf=evaluate_material(m, n, wo, m_wi);
      vec3 Le=evaluate_emissive(i, m_wi);
      result+=brdf*dot(m_wi, n)*mis_weight*Le/m_pdf;
    }
  }
  return result;
}
