#pragma once

#include "owl/owl.h"
#include "owl/common/math/vec.h"
#include "Renderer.h"
#include <cuda_runtime.h>

namespace cdf {
  
  using namespace owl;
  using namespace owl::common;
 
  struct RayGen {
  };

  struct TriangleGeom {
    vec3f *vertexBuffer;
    vec3i *indexBuffer;
  };

  struct CdfGeom {
    float *triPdfs;
    float geomPdf;
    int rowStart;
    int rowCount;
  };

  struct LaunchParams
  {
    uint32_t *fbPointer;
    float4   *accumBuffer;
    int       accumID;
    struct {
      vec3f org;
      vec3f dir_00;
      vec3f dir_du;
      vec3f dir_dv;
    } camera;
    // 3D model used in rendering mode
    struct {
      vec3f *vertexBuffer;
      vec3f *indexBuffer;
      OptixTraversableHandle group;
    } model;
    // Rendering
    int renderMode;
    // Benchmarks
    int benchmarkMode;
    int *offSamples; // [0]: off-by-one, [1]: off-by-two, ...
    int offSamplesMax;
    int *offLuminance; // 12 values, off by < 1e-30,1e-20,1e-15,1e-10,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,>=1e-2
    float luminanceMin;
    float luminanceMax;
    float hdriIntensity;
    // hdr texture
    cudaTextureObject_t hdrTexture;
    // conventional cdf sampling
    float *environmentMapRows;
    float *environmentMapCols;
    int environmentMapWidth;
    int environmentMapHeight;
    // cdf sampling w/ triangle bvh
    OptixTraversableHandle cdf;
    vec3f *vertexBuffer; // TODO: is TriangleGeom not available in raygen?
    vec3i *indexBuffer;
    struct {
      int   heatMapEnabled;
      float heatMapScale;
      int   spp;
    } render;
  };
  
}
