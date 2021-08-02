template<Render::Mode Mode>
inline __device__
void traverse(const VolumeGeomData& self, PerRayData& prd) {
  const vec3i& dims = self.dims;
  const float  ds   = 2.f*self.step;

  // Begin volume traversal event.
  TraversalData& tdata = prd.tdata;
  tdata.event = Interaction::Event::Traverse;

  vec3f& org = tdata.org;
  vec3f& dir = tdata.dir;

  while (self.bounds.contains(org)) {
    // Fetch gradient.
    const vec3i cell   = getCell(self, org);
    const vec3f weight = org - vec3f(cell);
    const vec3f grad   = fetchGradient(self.gradient, cell, dims, weight);

    // Store sample (if necessary).
    if (Mode == Render::Mode::Sample) {
      if (length(grad) > 0.f) {
        if (!storeSample(prd))
          return;
      }
    }

    // Step to next sample.
    const vec3f porg = org;

    float& eta      = tdata.eta;
    vec3f& atten    = tdata.atten;
    int&   ndata    = tdata.ndata;
    float& distance = tdata.distance;

    org += (ds/eta)*dir;
    dir += ds*grad;
    eta += dot(grad, org - porg);

    const float len = length(org - porg);
    distance += len;

    if (self.ior_mask == nullptr || getMaskValue(self, cell)) {
      atten *= attenuate(self.absorb, len);
      ++ndata;
    }
  }

  // End volume traversal event.
  tdata.event  = Interaction::Event::Exit;

  prd.out.status = Path::Status::Bounced;
  prd.out.org    = org;
  prd.out.dir    = normalize(dir);
  prd.out.atten  = self.albedo*tdata.atten;
}
