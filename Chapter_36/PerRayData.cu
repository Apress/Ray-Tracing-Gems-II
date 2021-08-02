struct PerRayData {
  TraversalData& tdata;
  Random&        random;

  vec3f atten;

  struct {
    Path::Status status;
    vec3f org;
    vec3f dir;
    vec3f atten;
  } out;
};
