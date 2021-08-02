template<Render::Mode Mode>
inline __device__
vec3f tracePath(const RayGenData& rgd, Ray& ray, PerRayData& prd) {
  TraversalData& tdata = prd.tdata;
  tdata.org = ray.origin;
  tdata.dir = ray.direction;

  vec3f& atten = prd.atten;
  int&   depth = tdata.depth;

  int MaxDepth = MAX_DEPTH;

  while (depth < MaxDepth) {
    if (Mode == Render::Mode::Sample) {
      if (!storeSample(prd))
        return vec3f(0.f);
    }

    owl::traceRay(rgd.world, ray, prd);

    if (prd.out.status == Path::Status::Cancelled ||
        prd.out.status == Path::Status::Postponed)
      return vec3f(0.f);
    else if (prd.out.status == Path::Status::Missed) {
      atten *= missColor(ray);
      return atten;
    }
    else if (prd.out.status == Path::Status::Bounced)
      atten *= prd.out.atten;

    // Trace another ray.
    const vec3f& org = prd.out.org;
    const vec3f  dir = normalize(prd.out.dir);

    ray = Ray(org, dir, T_MIN, T_MAX);

    tdata.org = org;
    tdata.dir = dir;

    prd.out.status = Path::Status::Invalid;

    ++depth;
  }

  return atten;
}
