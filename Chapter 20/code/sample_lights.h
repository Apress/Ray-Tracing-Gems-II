bool sample_lights(vec3 P, vec3 n, vec3 *P_out, vec3 *n_out,
  vec3 *Le_out, float *pdf_out)
{
  int chosen_light = floor(uniform()*NUM_LIGHTS);
  vec3 l_pos, l_nor;
  float p = 1.0 / NUM_LIGHTS;
  Object l = objects[chosen_light];
  
  if (l.type == GeometryType::Sphere) {
    float r = l.size.x;
    // Choose a normal on the side of the sphere visible to P.
    l_nor = random_hemisphere(P-l.pos);
    l_pos = l.pos + l_nor * r;
    float area_half_sphere = 2.0*PI*r*r;
    p /= area_half_sphere;
  } else if (l.type == GeometryType::Quad) {
    l_pos = l.pos + random_quad(l.normal, l.size);
    l_nor = l.normal;
    float area = l.size.x*l.size.y;
    p /= area;
  }

  bool vis = dot(P-l_pos, l_nor) > 0.0; // Light front side
  vis &= dot(P-l_pos, n) < 0.0; // Behind the surface at P
  // Shadow ray
  vis &= intersect_visibility(safe(P, n), safe(l_pos, l_nor));

  *P_out = l_pos;
  *n_out = l_nor;
  *pdf_out = p;
  *Le_out = vis ? l.material.emissive : vec3(0.0);
  return vis;
}
