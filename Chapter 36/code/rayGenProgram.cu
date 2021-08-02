OPTIX_RAYGEN_PROGRAM(rayGen)() {
  switch (optixLaunchParams.rmode) {
  case Render::Mode::Normal:
    rayGenFcn<Render::Mode::Normal>::run();
    break;

  case Render::Mode::Sample:
    rayGenFcn<Render::Mode::Sample>::run();
    break;
  }
}
