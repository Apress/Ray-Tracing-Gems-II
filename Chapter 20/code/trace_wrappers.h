// This workaround is to be able to reuse direct_mis.h to to debug images without changing it since it is part of the article
// Define two new functions that pretend sample_lights or sample_material says that it fails give only the other part
#define sample_lights false
#define direct_mis direct_mis_material
#include "direct_mis.h"
#undef sample_lights
#undef direct_mis
#define sample_material false
#define direct_mis direct_mis_light
#include "direct_mis.h"
#undef sample_material
#undef direct_mis

vec3 trace_normals(vec3 p, vec3 d) {
  Intersect i = intersect(p, d);
  if (i.hit) {
    return vec3(abs(i.nor.x), abs(i.nor.y), abs(i.nor.z));
  } else {
    return vec3(1.0, 0.0, 1.0);
  }
}

vec3 show_color(vec3 p, vec3 d) {
  Intersect i = intersect(p, d);
  if (i.hit) {
    return i.mat.diffuse;
  } else {
    return vec3(1.0, 0.0, 1.0);
  }
}

vec3 show_scene(vec3 p, vec3 d) {
  Intersect i = intersect(p, d);
  if (i.hit) {
    if (i.mat.is_emissive) {
      // Normalize emissive so we can understand color even if very bright
      vec3 e=i.mat.emissive;
      float a = max(max(e.x, e.y), e.z);
      if (a<1.0f) a = 1.0f;
      return i.mat.emissive * (1.0f/a);
    }
    return i.mat.roughness;
  } else {
    return vec3(0.0, 0.0, 0.0);
  }
}

vec3 trace_emissive(vec3 p, vec3 d) {
  Intersect i = intersect(p, d);
  if (i.hit) {
    return i.mat.emissive;
  } else {
    return vec3(1.0, 0.0, 1.0);
  }
}

vec3 trace_direct_cos_sampling(vec3 P, vec3 d) {
  Intersect i = intersect(P, d);
  if (!i.hit) return vec3(0.0);
  if (i.mat.is_emissive) return evaluate_emissive(i, d);
  return direct_cos(safe(i.pos, i.nor), i.nor, -d, i.mat);
}

vec3 trace_direct_material_sampling(vec3 P, vec3 d) {
  Intersect i = intersect(P, d);
  if (!i.hit) return vec3(0.0);
  if (i.mat.is_emissive) return evaluate_emissive(i, d);
  return direct_mat(safe(i.pos, i.nor), i.nor, -d, i.mat);
}

vec3 trace_direct_light_sampling(vec3 P, vec3 d) {
  Intersect i = intersect(P, d);
  if (!i.hit) return vec3(0.0);
  if (i.mat.is_emissive) return evaluate_emissive(i, d);
  return direct_light(safe(i.pos, i.nor), i.nor, -d, i.mat);
}

vec3 trace_direct_mis(vec3 P, vec3 d) {
  Intersect i = intersect(P, d);
  if (!i.hit) return vec3(0.0);
  if (i.mat.is_emissive) return evaluate_emissive(i, d);
  return direct_mis(safe(i.pos, i.nor), i.nor, -d, i.mat);
}

template<int BOOST = 1, bool INCLUDE_EMISSIVE = false>
vec3 trace_direct_mis_material(vec3 P, vec3 d) {
  Intersect i = intersect(P, d);
  if (!i.hit) return vec3(0.0);
  if (i.mat.is_emissive) {
    if (INCLUDE_EMISSIVE) return evaluate_emissive(i, d);
    return vec3(0.0f);
  }
  return direct_mis_material(safe(i.pos, i.nor), i.nor, -d, i.mat) * BOOST;
}

template<int BOOST = 1, bool INCLUDE_EMISSIVE = false>
vec3 trace_direct_mis_light(vec3 P, vec3 d) {
  Intersect i = intersect(P, d);
  if (!i.hit) return vec3(0.0);
  if (i.mat.is_emissive) {
    if (INCLUDE_EMISSIVE) return evaluate_emissive(i, d);
    return vec3(0.0f);
  }
  return direct_mis_light(safe(i.pos, i.nor), i.nor, -d, i.mat) * BOOST;
}

vec3 pathtrace_mis_helper(vec3 P, vec3 d) {
  Intersect i = intersect(P, d);
  if (!i.hit) return vec3(0.0);
  if (i.mat.is_emissive) return evaluate_emissive(i, d);
  return pathtrace_mis(safe(i.pos, i.nor), i.nor, -d, i.mat);
}
