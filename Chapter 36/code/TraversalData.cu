struct TraversalData {
  Random random;

  int pidx;
  int depth;

  Interaction::Event event;

  vec3f org;
  vec3f dir;
  vec3f atten;

  const VolumeGeomData* vptr;

  float eta;
  int   ndata;
  float distance;
};
