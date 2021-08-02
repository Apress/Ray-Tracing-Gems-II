inline __device__
bool storeSample(PerRayData& prd) {
  const int& MaxSampleDepth = optixLaunchParams.msd;

  const int& pidx     = prd.tdata.pidx;
        int& nsamples = optixLaunchParams.nsamplesBuffer[pidx];
  if (nsamples >= MaxSampleDepth) {
    prd.out.status = Path::Status::Postponed;
    optixLaunchParams.doneBuffer[pidx] = 0;

    return false;
  }

  // Store sample point.
  const vec2i&      fbSize = optixLaunchParams.fbSize;
  const int         sidx   = nsamples*fbSize.y*fbSize.x + pidx;
        SampleData& sample = optixLaunchParams.sdataBuffer[sidx];

  sample = SampleData(prd.tdata);
  ++nsamples;

  return true;
}
