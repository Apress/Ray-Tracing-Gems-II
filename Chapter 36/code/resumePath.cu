inline __device__
void resumePath(const RayGenData& rgd,
                      Ray&        ray,
                      PerRayData& prd) {
  if (prd.tdata.event == Interaction::Event::Exit) {
    // Postponed from previous traverse(...) call but after exiting
    //   volume, so just fall through to continue tracing.
  }
  else if (prd.tdata.event == Interaction::Event::Miss) {
    // Postponed when attempting to store path end point, so store 
    //   endpoint and return.
    storeSample(prd);
    return;
  }
  else if (prd.tdata.event == Interaction::Event::Traverse) {
    // Postponed mid-traversal from previous traverse(...) call, so
    //   resume traversal.
    const VolumeGeomData& volume = *(prd.tdata.vptr);
    traverse<Render::Mode::Sample>(volume, prd);

    if (prd.out.status == Path::Status::Postponed) {
      // Did not finish traversing volume, so return to caller
      //   (through rayGen()) for another pass.
      return;
    }

    // Finished traversing volume, so initialize next ray, increment
    //   depth, and ...
    ray = Ray(prd.tdata.org, prd.tdata.dir, T_MIN, T_MAX);
    prd.out.status = Path::Status::Invalid;
    ++prd.tdata.depth;

    // ... fall though to continue tracing.
  }

  // Continue propagation.
  tracePath<Render::Mode::Sample>(rgd, ray, prd);
}
