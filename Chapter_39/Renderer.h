#pragma once

#include "owl/owl.h"
#include "owl/common/math/box.h"
#include "owl/common/math/vec.h"
#include <cuda_runtime.h>

namespace cdf {

  using namespace owl;
  using namespace owl::common;

  enum RenderMode {
    RenderModeBinarySearch,
    RenderModeBVH,
    RenderModeRandom,
  };

  enum BenchmarkMode {
    BenchmarkModeBinarySearch,
    BenchmarkModeBVH,
    BenchmarkModeErrors,
    BenchmarkModeNone,
  };

  struct Renderer {
    Renderer(std::string hdrFileName, std::string objFileName);

    void setCamera(const vec3f &org,
                   const vec3f &dir_00,
                   const vec3f &dir_du,
                   const vec3f &dir_dv);
    void render(const vec2i &fbSize,
                uint32_t *fbPointer);
    void beginErrorBenchmark();
    void endErrorBenchmark();

    OWLParams  lp;
    OWLRayGen  rayGenRender;
    OWLRayGen  rayGenBenchmark;
    OWLContext owl;
    OWLModule  module;

    struct Texture {
      cudaArray_t pixelArray;
      cudaTextureObject_t texObj;
    };
    Texture hdrTexture;

    OWLBuffer environmentMapRows;
    OWLBuffer environmentMapCols;

    OWLBuffer offSamples;
    OWLBuffer offLuminance;

    OWLGeomType cdfGeomType;
    OWLGeomType modelGeomType;
    OWLGroup cdfGroup;
    OWLGroup modelGroup;

    OWLBuffer accumBuffer { 0 };
    int accumID { 0 };

    void resetAccum() { accumID = 0; }
    
    vec2i      fbSize { 1 };

    box3f modelBounds;
    
    static int   spp;
    static float hdriIntensity;
    static bool  heatMapEnabled;
    static float heatMapScale;
    static RenderMode renderMode;
    static BenchmarkMode benchmarkMode;
    static int offSamplesMax;
    static bool cdfDumpAsObj;
    static float simplificationRate;
    static bool terminateRenderer;
  };
  
} // ::cdf
