OPTIX_MISS_PROGRAM(miss)() {
  PerRayData& prd = owl::getPRD<PerRayData>();
  prd.out.status  = Path::Status::Missed;

  // Store path endpoint (if necessary).
  if (optixLaunchParams.rmode == Render::Mode::Sample) {
    const vec3f org = optixGetWorldRayOrigin();
          vec3f dir = optixGetWorldRayDirection();

    dir = normalize(dir);

    prd.tdata.org   = org + dir;
    prd.tdata.dir   = dir;
    prd.tdata.event = Interaction::Event::Miss;

    storeSample(prd);
  }
}
