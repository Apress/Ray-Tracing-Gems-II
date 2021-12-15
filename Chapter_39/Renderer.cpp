#include <cfloat>
#include <iomanip>
#include "cdf.h"
#include "Renderer.h"
#include "deviceCode.h"
#include "owl/helper/cuda.h"

namespace cdf {
  extern "C" char deviceCode_ptx[];

  bool  Renderer::terminateRenderer = false;
  bool  Renderer::heatMapEnabled = false;
  float Renderer::heatMapScale = 1e-5f;
  int   Renderer::spp = 4;
  float   Renderer::hdriIntensity = 1.f;
  RenderMode Renderer::renderMode = RenderModeBVH;
  BenchmarkMode Renderer::benchmarkMode = BenchmarkModeNone;
  int Renderer::offSamplesMax = 10;
  bool Renderer::cdfDumpAsObj = false;
  float Renderer::simplificationRate = .999f;
  
  OWLVarDecl rayGenVars[]
  = {
     { nullptr /* sentinel to mark end of list */ }
  };

  OWLVarDecl triangleGeomVars[]
  = {
     { "indexBuffer",  OWL_BUFPTR, OWL_OFFSETOF(TriangleGeom,indexBuffer)},
     { "vertexBuffer", OWL_BUFPTR, OWL_OFFSETOF(TriangleGeom,vertexBuffer)},
     { nullptr /* sentinel to mark end of list */ }
  };

  OWLVarDecl CdfGeomVars[]
  = {
     { "triPdfs",  OWL_BUFPTR, OWL_OFFSETOF(CdfGeom,triPdfs)},
     { "geomPdf", OWL_FLOAT, OWL_OFFSETOF(CdfGeom,geomPdf)},
     { "rowStart", OWL_INT, OWL_OFFSETOF(CdfGeom,rowStart)},
     { "rowCount", OWL_INT, OWL_OFFSETOF(CdfGeom,rowCount)},
     { nullptr /* sentinel to mark end of list */ }
  };

  OWLVarDecl launchParamsVars[]
  = {
     { "fbPointer",   OWL_RAW_POINTER, OWL_OFFSETOF(LaunchParams,fbPointer) },
     { "accumBuffer",   OWL_BUFPTR, OWL_OFFSETOF(LaunchParams,accumBuffer) },
     { "accumID",   OWL_INT, OWL_OFFSETOF(LaunchParams,accumID) },
     // hdr map
     { "hdrTexture", OWL_USER_TYPE(cudaTextureObject_t), OWL_OFFSETOF(LaunchParams,hdrTexture) },
     // cdf texture
     { "environmentMapRows", OWL_BUFPTR, OWL_OFFSETOF(LaunchParams,environmentMapRows) },
     { "environmentMapCols", OWL_BUFPTR, OWL_OFFSETOF(LaunchParams,environmentMapCols) },
     { "environmentMapWidth", OWL_INT, OWL_OFFSETOF(LaunchParams,environmentMapWidth) },
     { "environmentMapHeight", OWL_INT, OWL_OFFSETOF(LaunchParams,environmentMapHeight) },
     // Model, if in rendering mode
     { "model.group", OWL_GROUP,  OWL_OFFSETOF(LaunchParams,model.group)},
     { "model.indexBuffer",  OWL_BUFPTR, OWL_OFFSETOF(LaunchParams,model.indexBuffer)},
     { "model.vertexBuffer", OWL_BUFPTR, OWL_OFFSETOF(LaunchParams,model.vertexBuffer)},
     // Rendering
     { "renderMode", OWL_INT, OWL_OFFSETOF(LaunchParams,renderMode) },
     // benchmark mode
     { "benchmarkMode", OWL_INT, OWL_OFFSETOF(LaunchParams,benchmarkMode) },
     { "offSamples", OWL_BUFPTR, OWL_OFFSETOF(LaunchParams,offSamples) },
     { "offSamplesMax", OWL_INT, OWL_OFFSETOF(LaunchParams,offSamplesMax) },
     { "offLuminance", OWL_BUFPTR, OWL_OFFSETOF(LaunchParams,offLuminance) },
     { "luminanceMin", OWL_FLOAT, OWL_OFFSETOF(LaunchParams,luminanceMin) },
     { "luminanceMax", OWL_FLOAT, OWL_OFFSETOF(LaunchParams,luminanceMax) },
     { "hdriIntensity", OWL_FLOAT, OWL_OFFSETOF(LaunchParams,hdriIntensity) },
     // cdf triangle BVH
     { "cdf", OWL_GROUP,  OWL_OFFSETOF(LaunchParams,cdf)},
     // TODO: can we just use TriangleGeom in raygen?
     { "indexBuffer",  OWL_BUFPTR, OWL_OFFSETOF(LaunchParams,indexBuffer)},
     { "vertexBuffer", OWL_BUFPTR, OWL_OFFSETOF(LaunchParams,vertexBuffer)},
     // render settings
     { "render.spp",           OWL_INT,   OWL_OFFSETOF(LaunchParams,render.spp) },
     { "render.heatMapEnabled", OWL_INT, OWL_OFFSETOF(LaunchParams,render.heatMapEnabled) },
     { "render.heatMapScale", OWL_FLOAT, OWL_OFFSETOF(LaunchParams,render.heatMapScale) },
     // camera settings
     { "camera.org",    OWL_FLOAT3, OWL_OFFSETOF(LaunchParams,camera.org) },
     { "camera.dir_00", OWL_FLOAT3, OWL_OFFSETOF(LaunchParams,camera.dir_00) },
     { "camera.dir_du", OWL_FLOAT3, OWL_OFFSETOF(LaunchParams,camera.dir_du) },
     { "camera.dir_dv", OWL_FLOAT3, OWL_OFFSETOF(LaunchParams,camera.dir_dv) },
     { nullptr /* sentinel to mark end of list */ }
  };
 
  Renderer::Texture xxxTexture2DCreateFP(int width, int height, int numComponents,
                                         const float *pixel,
                                         cudaTextureFilterMode filterMode,
                                         cudaTextureAddressMode addressMode,
                                         cudaTextureReadMode readMode)
  {
    Renderer::Texture result;

    int pitch = width*4*sizeof(float);
    std::vector<float> pixelData(height*pitch);
    if (numComponents==4) {
      memcpy(pixelData.data(),pixel,height*pitch);
    } else if (numComponents==3) {
      for (int y=0; y<height; ++y) {
        for (int x=0; x<width; ++x) {
          const float *src = pixel + (y * width*3 + x*3);
          for (int c=0; c<3; ++c) {
            pixelData[y * width*4 + x*4 + c] = src[c];
          }
          pixelData[y * width*4 + x*4 + 3] = 1.f;
        }
      }
    } else { assert(0 && "Unsupported number of color components"); }

    cudaResourceDesc res_desc = {};
    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float4>();
    cudaArray_t &pixelArray = result.pixelArray;
    cudaMallocArray(&pixelArray,
                    &channel_desc,
                    width,height);
    cudaMemcpy2DToArray(pixelArray,
                        0,0,
                        pixelData.data(),
                        pitch,pitch,height,
                        cudaMemcpyHostToDevice);
    res_desc.resType          = cudaResourceTypeArray;
    res_desc.res.array.array  = pixelArray;
    
    cudaTextureDesc tex_desc     = {};
    tex_desc.addressMode[0]      = addressMode;
    tex_desc.addressMode[1]      = addressMode;
    tex_desc.filterMode          = filterMode;
    tex_desc.readMode            = readMode;
    tex_desc.normalizedCoords    = 0;
    tex_desc.maxAnisotropy       = 0;
    tex_desc.maxMipmapLevelClamp = 0;
    tex_desc.minMipmapLevelClamp = 0;
    tex_desc.mipmapFilterMode    = cudaFilterModePoint;
    tex_desc.borderColor[0]      = 0.f;
    tex_desc.sRGB                = 0;
    
    // Create texture object
    cudaTextureObject_t cuda_tex = 0;
    cudaCreateTextureObject(&cuda_tex, &res_desc, &tex_desc, nullptr);
    result.texObj = cuda_tex;

    return result;
  }

  Renderer::Renderer(std::string hdrFileName, std::string objFileName, std::string blueNoiseFileName)
  {
    CDF cdf(hdrFileName);

    owl = owlContextCreate(nullptr,1);
    module = owlModuleCreate(owl,deviceCode_ptx);
    rayGenRender = owlRayGenCreate(owl,module,"renderFrame",
                             sizeof(RayGen),rayGenVars,-1);
    rayGenBenchmark = owlRayGenCreate(owl,module,"benchmark",
                             sizeof(RayGen),rayGenVars,-1);
    lp = owlParamsCreate(owl,sizeof(LaunchParams),launchParamsVars,-1);

    // ------------------------------------------------------------------
    // Blue Noise Mask
    // ------------------------------------------------------------------


    // ------------------------------------------------------------------
    // HDRI
    // ------------------------------------------------------------------
    hdrTexture = xxxTexture2DCreateFP(cdf.hdri.width,
                                      cdf.hdri.height,
                                      cdf.hdri.numComponents,
                                      cdf.hdri.pixel.data(),
                                      cudaFilterModeLinear,
                                      cudaAddressModeClamp,
                                      cudaReadModeElementType);
    owlParamsSetRaw(lp,"hdrTexture",&hdrTexture.texObj);
    float luminanceMin =  FLT_MAX;
    float luminanceMax = -FLT_MAX;
    for (int y=0; y<cdf.hdri.height; ++y) {
      for (int x=0; x<cdf.hdri.width; ++x) {
        const float *sample=cdf.hdri.samplePoint(x,y);
        luminanceMin = std::min(luminanceMin,std::min(sample[0],std::min(sample[1],sample[2])));
        luminanceMax = std::max(luminanceMax,std::max(sample[0],std::max(sample[1],sample[2])));
      }
    }
    owlParamsSet1f(lp,"luminanceMin",luminanceMin);
    owlParamsSet1f(lp,"luminanceMax",luminanceMax);

    // ------------------------------------------------------------------
    // CDF
    // ------------------------------------------------------------------
    environmentMapRows = owlDeviceBufferCreate(owl,OWL_FLOAT,cdf.hdri.width,cdf.cumulatedLastCol.data());
    environmentMapCols = owlDeviceBufferCreate(owl,OWL_FLOAT,cdf.hdri.width*cdf.hdri.height,cdf.cumulatedRows.data());
    owlParamsSetBuffer(lp,"environmentMapRows",environmentMapRows);
    owlParamsSetBuffer(lp,"environmentMapCols",environmentMapCols);
    owlParamsSet1i(lp,"environmentMapWidth",cdf.hdri.width);
    owlParamsSet1i(lp,"environmentMapHeight",cdf.hdri.height);

    // ------------------------------------------------------------------
    // Triangle CDF
    // ------------------------------------------------------------------
    cdfGeomType = owlGeomTypeCreate(owl,
                                         OWL_TRIANGLES,
                                         sizeof(CdfGeom),
                                         CdfGeomVars, -1);

    owlGeomTypeSetClosestHit(cdfGeomType, 0, module, "CdfCH");

    // ------------------------------------------------------------------
    // Model geom type
    // ------------------------------------------------------------------
    modelGeomType = owlGeomTypeCreate(owl,
                                      OWL_TRIANGLES,
                                      sizeof(TriangleGeom),
                                      triangleGeomVars, -1);

    owlGeomTypeSetClosestHit(modelGeomType, 0, module, "ModelCH");

    Mesh::Representation mode = Mesh::Ribbons;

    Mesh mesh = cdf.asTriangleMesh(mode, simplificationRate, cdfDumpAsObj);

    // Just lazily add any potential obj model on top....
    if (!objFileName.empty()) {
      modelBounds = { vec3f(FLT_MAX), vec3f(-FLT_MAX) };
      try {
        Mesh objMesh = Mesh(objFileName);
        // Construct bounds
        for (std::size_t i=0; i<objMesh.geoms.size(); ++i)
        {
          const Geometry &geom = objMesh.geoms[i];
          for (const auto &v : geom.vertex) {
            modelBounds.extend(v);
          }
        }
        // Push onto geometry stack
        for (std::size_t i=0; i<objMesh.geoms.size(); ++i)
        {
          Geometry &geom = objMesh.geoms[i];
          if (i==0) {
            // Add a ground plane to the first mesh
            vec3f v1(modelBounds.lower.x*100.f,modelBounds.lower.y,modelBounds.lower.z*100.f);
            vec3f v2(modelBounds.upper.x*100.f,modelBounds.lower.y,modelBounds.lower.z*100.f);
            vec3f v3(modelBounds.upper.x*100.f,modelBounds.lower.y,modelBounds.upper.z*100.f);
            vec3f v4(modelBounds.lower.x*100.f,modelBounds.lower.y,modelBounds.upper.z*100.f);
            vec3i idx1 = vec3i(0,1,2)+vec3i((int)geom.vertex.size());
            vec3i idx2 = vec3i(0,2,3)+vec3i((int)geom.vertex.size());
            geom.vertex.push_back(v1);
            geom.vertex.push_back(v2);
            geom.vertex.push_back(v3);
            geom.vertex.push_back(v4);
            geom.index.push_back(idx1);
            geom.index.push_back(idx2);
          }
          mesh.geoms.push_back(geom);
        }
      } catch (const std::exception& ex) { std::cerr << "Cannot load... " << ex.what() << "\n"; }
    }


    size_t numNonEmptyRows = 0;
    size_t numQuads = 0;
    size_t numModelGeoms = 0;
    for (Geometry &geom : mesh.geoms) {
      if (geom.tag == Geometry::CDF && (!(geom.vertex.empty() || geom.index.empty())))
        numNonEmptyRows++;
      else if (geom.tag == Geometry::Model)
        numModelGeoms++;
      else {
        throw std::runtime_error("Unsupported geometry type");
      }
    }

    cdfGroup = owlInstanceGroupCreate(owl,1);
    if (numModelGeoms > 0)
      modelGroup = owlInstanceGroupCreate(owl,numModelGeoms);

    unsigned nonEmptyRowIndex = 0;
    unsigned modelIndex = 0;

    std::vector<OWLGeom> cdfGeoms;
    for (std::size_t i = 0; i < mesh.geoms.size(); ++i) {
      const Geometry &geom = mesh.geoms[i];

      // Some CDF rows are merged together to improve performance.
      // Removed rows will be marked as CDF geometry, and have no tris.
      if (geom.vertex.empty() || geom.index.empty()) {
        assert(geom.tag == Geometry::CDF);
        continue;
      }

      OWLBuffer indexBuffer = owlDeviceBufferCreate(owl, OWL_INT3, geom.index.size(), geom.index.data());
      OWLBuffer vertexBuffer = owlDeviceBufferCreate(owl, OWL_FLOAT3, geom.vertex.size(), geom.vertex.data());

      OWLGeom ogeom;
      if (geom.tag == Geometry::Model)
        ogeom = owlGeomCreate(owl, modelGeomType); // different CH..
      else
        ogeom = owlGeomCreate(owl, cdfGeomType);
      owlTrianglesSetIndices(ogeom, indexBuffer, geom.index.size(), sizeof(vec3i), 0);
      owlTrianglesSetVertices(ogeom, vertexBuffer, geom.vertex.size(), sizeof(vec3f), 0);
            
      if (geom.tag == Geometry::Model) {
        OWLGroup gi = owlTrianglesGeomGroupCreate(owl, 1, &ogeom);
        owlGroupBuildAccel(gi);
        owlParamsSetBuffer(lp,"model.vertexBuffer",vertexBuffer);
        owlParamsSetBuffer(lp,"model.indexBuffer",indexBuffer);
        owlGeomSetBuffer(ogeom,"vertexBuffer",vertexBuffer);
        owlGeomSetBuffer(ogeom,"indexBuffer",indexBuffer);
        owlInstanceGroupSetChild(modelGroup, modelIndex, gi);
        modelIndex++;
      } else {
        OWLBuffer triPdfBuffer = owlDeviceBufferCreate(owl, OWL_FLOAT, geom.triPdfs.size(), geom.triPdfs.data());
        owlGeomSetBuffer(ogeom, "triPdfs", triPdfBuffer);
        owlGeomSet1f(ogeom, "geomPdf", geom.geomPdf);
        owlGeomSet1i(ogeom, "rowStart", geom.rowStart);
        owlGeomSet1i(ogeom, "rowCount", geom.rowCount);
        nonEmptyRowIndex++;
      }
      cdfGeoms.push_back(ogeom);
    }

    OWLGroup gi = owlTrianglesGeomGroupCreate(owl, numNonEmptyRows, cdfGeoms.data());
    owlGroupBuildAccel(gi);
    owlInstanceGroupSetChild(cdfGroup, 0, gi);

    owlGroupBuildAccel(cdfGroup);
    owlParamsSetGroup(lp,"cdf",cdfGroup);

    if (numModelGeoms > 0) {
      owlGroupBuildAccel(modelGroup);
      owlParamsSetGroup(lp,"model.group",modelGroup);
    }

    owlBuildPrograms(owl);
    owlBuildPipeline(owl);
    owlBuildSBT(owl);
  }

  void Renderer::setCamera(const vec3f &org,
                           const vec3f &dir_00,
                           const vec3f &dir_du,
                           const vec3f &dir_dv)
  {
    owlParamsSet3f(lp,"camera.org",   org.x,org.y,org.z);
    owlParamsSet3f(lp,"camera.dir_00",dir_00.x,dir_00.y,dir_00.z);
    owlParamsSet3f(lp,"camera.dir_du",dir_du.x,dir_du.y,dir_du.z);
    owlParamsSet3f(lp,"camera.dir_dv",dir_dv.x,dir_dv.y,dir_dv.z);
  }

  void Renderer::render(const vec2i &fbSize,
                        uint32_t *fbPointer)
  {
    if (fbSize != this->fbSize) {
      if (!accumBuffer)
        accumBuffer = owlDeviceBufferCreate(owl,OWL_FLOAT4,1,nullptr);
      owlBufferResize(accumBuffer,fbSize.x*fbSize.y);
      owlParamsSetBuffer(lp,"accumBuffer",accumBuffer);
      this->fbSize = fbSize;
    }
    owlParamsSetPointer(lp,"fbPointer",fbPointer);

    owlParamsSet1i(lp,"accumID",accumID);
    accumID++;
    owlParamsSet1i(lp,"render.spp",max(spp,1));
    owlParamsSet1i(lp,"render.heatMapEnabled",heatMapEnabled);
    owlParamsSet1f(lp,"render.heatMapScale",heatMapScale);
    owlParamsSet1i(lp,"renderMode",(int)renderMode);
    owlParamsSet1i(lp,"benchmarkMode",(int)benchmarkMode);
    owlParamsSet1f(lp,"hdriIntensity",hdriIntensity);
    
    if (terminateRenderer) {
      static int countdown = 100;
      countdown--;
      if (countdown < 0) {
        exit(0);
      }
    }

    if (benchmarkMode == BenchmarkModeNone)
      owlLaunch2D(rayGenRender,fbSize.x,fbSize.y,lp);
    else {
      beginErrorBenchmark();
      owlLaunch2D(rayGenBenchmark,fbSize.x,fbSize.y,lp);
      endErrorBenchmark();
    }
  }

  void Renderer::beginErrorBenchmark()
  {
    if (benchmarkMode != BenchmarkModeErrors)
      return;

    std::vector<int> zeros(max(offSamplesMax,12),0);
    offSamples = owlDeviceBufferCreate(owl,OWL_INT,zeros.size(),zeros.data());
    offLuminance = owlDeviceBufferCreate(owl,OWL_INT,zeros.size(),zeros.data());
    owlParamsSetBuffer(lp,"offSamples",offSamples);
    owlParamsSet1i(lp,"offSamplesMax",offSamplesMax);
    owlParamsSetBuffer(lp,"offLuminance",offLuminance);
  }

  void Renderer::endErrorBenchmark()
  {
    if (benchmarkMode != BenchmarkModeErrors)
      return;

    static int counter = 0;
    if (counter++ % 10 != 0) // that's to match the benchmark printf's in viewer.cpp
      return;

    int deviceID=0; // FIXME: multi GPU..
    const int *offSamplesPointer = (const int *)owlBufferGetPointer(offSamples,deviceID);
    std::vector<int> offSamplesHost(offSamplesMax);
    cudaMemcpy(offSamplesHost.data(),offSamplesPointer,offSamplesMax*sizeof(int),cudaMemcpyDeviceToHost);

    const int *offLuminancePointer = (const int *)owlBufferGetPointer(offLuminance,deviceID);
    std::vector<int> offLuminanceHost(12);
    cudaMemcpy(offLuminanceHost.data(),offLuminancePointer,12*sizeof(int),cudaMemcpyDeviceToHost);

    uint64_t samplesTotal = uint64_t(fbSize.x)*fbSize.y*spp*2;
    uint64_t offSamplesTotal = 0;
    for (int i=0; i<offSamplesMax; ++i) {
      offSamplesTotal += (uint64_t)offSamplesHost[i];
    }

    std::cout << std::fixed;
    std::cout << std::setprecision(2);
    std::cout << offSamplesTotal << " samples out of " << samplesTotal
              << " (" << double(offSamplesTotal)/samplesTotal*100. << "%) are off\n";

    for (int i=0; i<offSamplesMax; ++i) {
      double percentOff = double(offSamplesHost[i])/offSamplesTotal*100.;
      double percentTotal = double(offSamplesHost[i])/samplesTotal*100.;
      if (i == offSamplesMax-1)
        std::cout << "Samples that are off by " << (i+1) << " or more: " << offSamplesHost[i]
                  << " (" << percentOff << "% of off / " << percentTotal << "% of total)\n";
      else
        std::cout << "Samples that are off by " << (i+1) << ": " << offSamplesHost[i]
                  << " (" << percentOff << "% of off / " << percentTotal << "% of total)\n";
    }

    std::cout << "HDRI sample error < 1e-30: " << offLuminanceHost[ 0] << " (" << offLuminanceHost[ 0] / double(offSamplesTotal) * 100. << "%)\n";
    std::cout << "HDRI sample error < 1e-20: " << offLuminanceHost[ 1] << " (" << offLuminanceHost[ 1] / double(offSamplesTotal) * 100. << "%)\n";
    std::cout << "HDRI sample error < 1e-15: " << offLuminanceHost[ 2] << " (" << offLuminanceHost[ 2] / double(offSamplesTotal) * 100. << "%)\n";
    std::cout << "HDRI sample error < 1e-10: " << offLuminanceHost[ 3] << " (" << offLuminanceHost[ 3] / double(offSamplesTotal) * 100. << "%)\n";
    std::cout << "HDRI sample error < 1e-8 : " << offLuminanceHost[ 4] << " (" << offLuminanceHost[ 4] / double(offSamplesTotal) * 100. << "%)\n";
    std::cout << "HDRI sample error < 1e-7 : " << offLuminanceHost[ 5] << " (" << offLuminanceHost[ 5] / double(offSamplesTotal) * 100. << "%)\n";
    std::cout << "HDRI sample error < 1e-6 : " << offLuminanceHost[ 6] << " (" << offLuminanceHost[ 6] / double(offSamplesTotal) * 100. << "%)\n";
    std::cout << "HDRI sample error < 1e-5 : " << offLuminanceHost[ 7] << " (" << offLuminanceHost[ 7] / double(offSamplesTotal) * 100. << "%)\n";
    std::cout << "HDRI sample error < 1e-4 : " << offLuminanceHost[ 8] << " (" << offLuminanceHost[ 8] / double(offSamplesTotal) * 100. << "%)\n";
    std::cout << "HDRI sample error < 1e-3 : " << offLuminanceHost[ 9] << " (" << offLuminanceHost[ 9] / double(offSamplesTotal) * 100. << "%)\n";
    std::cout << "HDRI sample error < 1e-2 : " << offLuminanceHost[10] << " (" << offLuminanceHost[10] / double(offSamplesTotal) * 100. << "%)\n";
    std::cout << "HDRI sample error >= 1e-2: " << offLuminanceHost[11] << " (" << offLuminanceHost[11] / double(offSamplesTotal) * 100. << "%)\n";
  }

}
