OPTIX_CLOSEST_HIT_PROGRAM(Volume)() {
  PerRayData& prd = owl::getPRD<PerRayData>();

  switch (optixLaunchParams.rmode) {
  case Render::Mode::Normal:
    intersect<Render::Mode::Normal>(prd);
    break;

  case Render::Mode::Sample:
    intersect<Render::Mode::Sample>(prd);
    break;
  }
}

template<Render::Mode Mode>
inline __device__
void intersect(PerRayData& prd) {
  // Compute hit point.
  const vec3f org  = optixGetWorldRayOrigin();
  const vec3f dir  = optixGetWorldRayDirection();
  const float thit = optixGetRayTmax();
  const vec3f hitP = org + thit*dir;

  // Prepare traversal data.
  const VolumeGeomData& self  = owl::getProgramData<VolumeGeomData>();
        TraversalData&  tdata = prd.tdata;

  tdata.org  = hitP + (1e-6f)*dir;
  tdata.dir  = dir;
  tdata.vptr = &self;
  tdata.eta  = self.eta0;

  // Traverse volume.
  traverse<Mode>(self, prd);
}
