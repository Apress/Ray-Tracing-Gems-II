float sample_lights_pdf(vec3 P, vec3 n, Intersect emissive_surface) {
  Object object = objects[emissive_surface.object_index];
  if (!object.material.is_emissive) return 0.0;
  float p = 1.0/NUM_LIGHTS;
  if (object.type == GeometryType::Sphere) {
    float r = object.size.x;
    float area_half_sphere = 2.0*PI*r*r;
    p /= area_half_sphere;
  } else if (object.type == GeometryType::Quad) {
    float area_quad = object.size.x * object.size.y;
    p /= area_quad;
  }
  return p;
}