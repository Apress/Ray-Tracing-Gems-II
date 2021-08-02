class rayGenFcn<Render::Mode::Sample> {
public:

  static __device__ void run() {
    const RayGenData& self    = owl::getProgramData<RayGenData>();
    const vec2i       pixelID = owl::getLaunchIndex();

    const vec2i& fbSize  = optixLaunchParams.fbSize;
    const int    pidx    = pixelID.y*fbSize.x + pixelID.x;

    const int& pass = optixLaunchParams.pass;
    if (pass == 0) {
      int&           nsamples = optixLaunchParams.nsamplesBuffer[pidx];
      int&           done     = optixLaunchParams.doneBuffer    [pidx];
      TraversalData& tdata    = optixLaunchParams.tdataBuffer   [pidx];
      PerRayData     prd(tdata);

      // Clear values and ...
      tdata = TraversalData(pidx);
      tdata.random.init(pixelID.x, pixelID.y);

      nsamples = 0;
      done     = 1;

      // ... trace ray.
      const vec2f screen = (vec2f(pixelID) +
                            vec2f(0.5f, 0.5f))/vec2f(fbSize);

      const vec3f org = self.camera.origin;
      const vec3f dir
        = self.camera.lower_left_corner
        + screen.u*self.camera.horizontal
        + screen.v*self.camera.vertical
        - self.camera.origin;

      Ray ray(org, normalize(dir), T_MIN, T_MAX);
      tracePath<Render::Mode::Sample>(self, ray, prd);
    }
    else {
      int& done     = optixLaunchParams.doneBuffer    [pidx];
      int& nsamples = optixLaunchParams.nsamplesBuffer[pidx];
      if (done) {
        nsamples = 0;
        return;
      }

      // Load traversal data.
      TraversalData& tdata = optixLaunchParams.tdataBuffer[pidx];

      Ray        ray(tdata.org, tdata.dir, T_MIN, T_MAX);
      PerRayData prd(tdata);

      // Clear values and ...
      nsamples = 0;
      done     = 1;

      // ... resume tracing.
      resumePath(self, ray, prd);
    }
  }
};
