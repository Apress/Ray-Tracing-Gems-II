template <vec3 (*DIRECT_TECHNIQUE)(vec3 pos, vec3 nor, vec3 wo, Material mat)>
vec3 pathtrace(vec3 P, vec3 d) {
  vec3 result = vec3(0), tp = vec3(1);
  for (int bounce = 0;; bounce++) {
    Intersect i = intersect(P, d);
    if (!i.hit) break;

    if (i.mat.is_emissive) {
      if (bounce == 0) result += tp * i.mat.emissive;
      break;
    }

    result += tp * DIRECT_TECHNIQUE(safe(i.pos, i.nor), i.nor, -d, i.mat);

    // Bounce (note secondary ray not reused from material sampling in DIRECT_TECHNIQUE
    vec3 wi;
    float m_pdf;
    if (!sample_material(i.mat, i.nor, -d, &wi, &m_pdf)) {
      break;
    }
    tp *= evaluate_material(i.mat, i.nor, -d, wi) * dot(i.nor, wi) / m_pdf;

    if (russian_roulette(&tp)) break;

    d = wi;
    P = safe(i.pos, i.nor);
  }
  return result;
}
