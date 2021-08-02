struct VolumeGeomData {
  box3f bounds;
  vec3i dims;

  vec3f* vertex;
  vec3i* index;

  vec3f* gradient;
  int*   ior_mask;
};
