vec3 pathtrace_mis(vec3 P, vec3 n, vec3 wo, Material m) {
  vec3 result = vec3(0), tp = vec3(1);
  while (true) {
    vec3 Le, m_wi, l_pos, l_nor;
    float l_pdf, m_pdf;
    // Light sampling
    if (sample_lights(P, n, &l_pos, &l_nor, &Le, &l_pdf)) {
      vec3 l_wi = normalize(l_pos - P);
      float G=geom_fact_sa(P, l_pos, l_nor);
      float m_pdf=sample_material_pdf(m, n, wo, l_wi);
      float mis_weight=balance_heuristic(l_pdf, m_pdf*G);
      vec3 brdf=evaluate_material(m, n, wo, l_wi);
      result+=tp*brdf*G*clamped_dot(n, l_wi)*mis_weight*Le/l_pdf;
    }
    // For material sampling and bounce
    if (!sample_material(m, n, wo, &m_wi, &m_pdf)) {
      break;
    }
    Intersect i = intersect(safe(P, n), m_wi);
    if (!i.hit) {
      break; // Missed scene
    }
    tp*=evaluate_material(m, n, wo, m_wi)*dot(m_wi, n)/m_pdf;
    if (i.mat.is_emissive) {
      float G = geom_fact_sa(P, i.pos, i.nor);
      float light_pdf=sample_lights_pdf(P, n, i);
      float mis_weight=balance_heuristic(m_pdf*G, light_pdf);
      vec3 Le = evaluate_emissive(i, m_wi);
      result+=tp*mis_weight*Le;
      break; // Our emissive surface doesn't bounce.
    }

    if (russian_roulette(&tp)) break;

    // Update state for next bounce; tp captures material and pdf.
    P = i.pos;
    n = i.nor;
    wo = -m_wi;
    m = i.mat;
  }
  return result;
}
