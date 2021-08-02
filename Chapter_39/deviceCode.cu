// Use heights as pdf values
#define USE_HEIGHTS

#include "deviceCode.h"
#include "owl/common/math/random.h"

namespace cdf {

  extern "C" __constant__ LaunchParams optixLaunchParams;

  typedef owl::common::LCG<4> Random;
  
  inline __device__
  vec3f backGroundColor()
  {
    const vec2i pixelID = owl::getLaunchIndex();
    const float t = pixelID.y / (float)optixGetLaunchDimensions().y;
    const vec3f c = (1.0f - t)*vec3f(1.0f, 1.0f, 1.0f) + t * vec3f(0.5f, 0.7f, 1.0f);
    return c;
  }

  __device__ float linear_to_srgb(float x) {
      if (x <= 0.0031308f) {
          return 12.92f * x;
      }
      return 1.055f * pow(x, 1.f/2.4f) - 0.055f;
  }

  inline __device__ vec4f over(const vec4f &A, const vec4f &B)
  {
    return A + (1.f-A.w)*B;
  }


  // ==================================================================
  // conventional cdf sampling
  // ==================================================================

  __device__
  const float* upper_bound (const float* first, const float* last, const float& val)
  {
    const float* it;
  //   iterator_traits<const float*>::difference_type count, step;
    int count, step;
  //   count = std::distance(first,last);
    count = (last-first);
    while (count > 0)
    {
      it = first; 
      step=count/2; 
      // std::advance (it,step);
      it = it + step;
      if ( ! (val < *it))                 // or: if (!comp(val,*it)), for version (2)
      { 
          first=++it; 
          count-=step+1;  
      }
      else count=step;
    }
    return first;
  }
 
  struct Sample {
    int x; // column
    int y; // row
    float pdfx;
    float pdfy;
  };

  __device__ float sample_cdf(const float* data, unsigned int n, float x, unsigned int *idx, float* pdf) 
  {
      *idx = upper_bound(data, data + n, x) - data;
      float scaled_sample;
      if (*idx == 0) {
          *pdf = data[0];
          scaled_sample = x / data[0];
      } else {
          if (*idx < n) {
          *pdf = data[*idx] - data[*idx - 1];
          scaled_sample = (x - data[*idx - 1]) / (data[*idx] - data[*idx - 1]);
          } else { /*printf("Oups %i\n",*idx);*/ }
      }
      // keep result in [0,1)
      return min(scaled_sample, 0.99999994f);
  }

  // Uv range: [0, 1]
  __device__
  vec3f toPolar(vec2f uv)
  {
      float theta = 2.0 * M_PI * uv.x + - M_PI / 2.0;
      float phi = M_PI * uv.y;

      vec3f n;
      n.x = __cosf(theta) * __sinf(phi);
      n.z = __sinf(theta) * __sinf(phi);
      n.y = __cosf(phi);

      n.x = -n.x;
      return n;
  }

  __device__ Sample sampleCDF(float rx, float ry)
  {
    auto &lp = optixLaunchParams;
    float* rows = lp.environmentMapRows;
    float* cols = lp.environmentMapCols;
    int width = lp.environmentMapWidth;
    int height = lp.environmentMapHeight;
    float row_pdf, col_pdf;
    unsigned x, y;
    ry = sample_cdf(rows, height, ry, &y, &row_pdf);
    y = max(min(y, height - 1), 0);
    rx = sample_cdf(cols + y * width, width, rx, &x, &col_pdf);
    return {x,y,col_pdf,row_pdf}; // TODO: actual *light* sampling
  }

  struct LightSample {
    vec3f L;
    vec3f intensity;
    float pdf;
  };

  __device__
  vec2f toUV(vec3f n)
  {
    vec2f uv;
  
    uv.x = atan2f(float(n.x), float(n.z));
    uv.x = (uv.x + M_PI / 2.0f) / (M_PI * 2.0f) + M_PI * (28.670f / 360.0f);
  
    uv.y = clamp(float(acosf(n.y) / M_PI), .001f, .999f);
  
    return uv;
  }

  __device__  LightSample sampleEnvironmentMap(vec3f dir)
  {
    vec2f uv = toUV(dir);
    float u = uv.x;
    float v = uv.y;
    auto &lp = optixLaunchParams;
    vec4f texel = tex2D<float4>(lp.hdrTexture,u*(lp.environmentMapWidth-1),v*(lp.environmentMapHeight-1));
    return {dir,{texel.x,texel.y,texel.z}, 1.f/float(M_PI)};
  }

  __device__ vec3f cosineSampleHemisphere(float u1, float u2)
  {
    float r     = sqrtf(u1);
    float theta = u2 * 2.f * float(M_PI);
    float x     = r * cosf(theta);
    float y     = r * sinf(theta);
    float z     = sqrtf(1.f - u1);
    return {x,y,z};
  }


  // ==================================================================
  // Triangle model
  // ==================================================================

  struct ModelPRD {
    float t_hit;
    vec3f gn;
    int primID;
  };
 
  OPTIX_CLOSEST_HIT_PROGRAM(ModelCH)()
  {
    ModelPRD& prd = owl::getPRD<ModelPRD>();
    const TriangleGeom& self = owl::getProgramData<TriangleGeom>();
    prd.t_hit = optixGetRayTmax();
    prd.primID = optixGetPrimitiveIndex();
    const vec3i index  = self.indexBuffer[prd.primID];
    const vec3f& v1     = self.vertexBuffer[index.x];
    const vec3f& v2     = self.vertexBuffer[index.y];
    const vec3f& v3     = self.vertexBuffer[index.z];
    prd.gn            = normalize(cross(v2 - v1, v3 - v1));
  }

  // ==================================================================
  // cdf sampling w/ triangle BVH
  // ==================================================================

  struct PRD {
    float x;
    float y;
    float rowPdf;
    float colPdf;
  };
 
  static __forceinline__ __device__
  void *unpackPointer( uint32_t i0, uint32_t i1 )
  {
      const uint64_t uptr = static_cast<uint64_t>( i0 ) << 32 | i1;
      void*           ptr = reinterpret_cast<void*>( uptr ); 
      return ptr;
  }

  static __forceinline__ __device__
  void  packPointer( void* ptr, uint32_t& i0, uint32_t& i1 )
  {
      const uint64_t uptr = reinterpret_cast<uint64_t>( ptr );
      i0 = uptr >> 32;
      i1 = uptr & 0x00000000ffffffff;
  }

  template<typename T>
  static __forceinline__ __device__ T *getPRD()
  { 
      const uint32_t u0 = optixGetPayload_0();
      const uint32_t u1 = optixGetPayload_1();
      return reinterpret_cast<T*>( unpackPointer( u0, u1 ) );
  }

  OPTIX_CLOSEST_HIT_PROGRAM(CdfCH)()
  {
    const auto &self = owl::getProgramData<CdfGeom>();
    float2 b = optixGetTriangleBarycentrics();
    int id = optixGetPrimitiveIndex();
    int start = self.rowStart;
    int end = self.rowStart + (self.rowCount - 1);
    float alpha = ((id % 2) == 0) ? (b.x + b.y) : 1.f - (b.x + b.y);
    optixSetPayload_0(__float_as_int(optixGetRayTmax()));
    optixSetPayload_1(__float_as_int(start * (1.f - alpha) + end * alpha));
    optixSetPayload_2(__float_as_int(self.triPdfs[id]));
    optixSetPayload_3(__float_as_int(self.geomPdf));
  }

  __device__ Sample sampleCDF_BVH(float rx, float ry)
  {
    auto &lp = optixLaunchParams;    
    unsigned int p0, p1, p2, p3;
    optixTrace(lp.cdf,
      make_float3(rx, ry, 0.f),
      make_float3(0.f, 0.f, 1.f),
      0.f,    // tmin
      1.1f,  // tmax
      0.0f,   // rayTime
      OptixVisibilityMask( 255 ),
      OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
      0,// SBT offset
      1,// SBT stride
      0,// missSBTIndex 
      p0, p1, p2, p3);
    return {__int_as_float(p0) * lp.environmentMapWidth, __int_as_float(p1), __int_as_float(p2), __int_as_float(p3)};
  }


  // ==================================================================
  // Importance sampling interface
  // ==================================================================

  __device__ LightSample importanceSampleEnvironmentMap(float rx, float ry)
  {
    auto &lp = optixLaunchParams;
    int width = lp.environmentMapWidth;
    int height = lp.environmentMapHeight;
    Sample sample = sampleCDF(rx,ry);
    float invjacobian = width * height / float(4 * M_PI);
    vec3f L(toPolar(vec2f((sample.x) / float(width), (sample.y)/float(height))));
    LightSample ls = sampleEnvironmentMap(L);
    ls.pdf = sample.pdfx * sample.pdfy * invjacobian;
    return ls;
  }

  __device__ LightSample importanceSampleEnvironmentMapBVH(float rx, float ry)
  {
    auto &lp = optixLaunchParams;
    int width = lp.environmentMapWidth;
    int height = lp.environmentMapHeight;
    Sample sample = sampleCDF_BVH(rx,ry);
    float invjacobian = width * height / float(4 * M_PI);
    vec3f L(toPolar(vec2f((sample.x) / float(width), (sample.y)/float(height))));
    LightSample ls = sampleEnvironmentMap(L);
    ls.pdf = sample.pdfx * sample.pdfy * invjacobian;
    return ls;
  }

  __device__ LightSample importanceSampleEnvironmentMapRandom(float rx, float ry)
  {
    auto &lp = optixLaunchParams;
    vec3f L(toPolar(vec2f(rx, ry)));
    LightSample ls = sampleEnvironmentMap(L);
    ls.pdf = 1.f / float(4 * M_PI);
    return ls;
  }

  // ==================================================================
  //
  // ==================================================================

  inline __device__ Ray generateRay(const vec2f screen)
  {
    auto &lp = optixLaunchParams;
    vec3f org = lp.camera.org;
    vec3f dir
      = lp.camera.dir_00
      + screen.u * lp.camera.dir_du
      + screen.v * lp.camera.dir_dv;
    dir = normalize(dir);
    if (fabs(dir.x) < 1e-5f) dir.x = 1e-5f;
    if (fabs(dir.y) < 1e-5f) dir.y = 1e-5f;
    if (fabs(dir.z) < 1e-5f) dir.z = 1e-5f;
    return Ray(org,dir,0.f,1e10f);
  }
  
  inline __device__ vec3f hue_to_rgb(float hue)
  {
    float s = saturate( hue ) * 6.0f;
    float r = saturate( fabsf(s - 3.f) - 1.0f );
    float g = saturate( 2.0f - fabsf(s - 2.0f) );
    float b = saturate( 2.0f - fabsf(s - 4.0f) );
    return vec3f(r, g, b); 
  }
    
  inline __device__ vec3f temperature_to_rgb(float t)
  {
    float K = 4.0f / 6.0f;
    float h = K - K * t;
    float v = .5f + 0.5f * t;    return v * hue_to_rgb(h);
  }
    
                                    
  inline __device__
  vec3f heatMap(float t)
  {
#if 1
    return temperature_to_rgb(t);
#else
    if (t < .25f) return lerp(vec3f(0.f,1.f,0.f),vec3f(0.f,1.f,1.f),(t-0.f)/.25f);
    if (t < .5f)  return lerp(vec3f(0.f,1.f,1.f),vec3f(0.f,0.f,1.f),(t-.25f)/.25f);
    if (t < .75f) return lerp(vec3f(0.f,0.f,1.f),vec3f(1.f,1.f,1.f),(t-.5f)/.25f);
    if (t < 1.f)  return lerp(vec3f(1.f,1.f,1.f),vec3f(1.f,0.f,0.f),(t-.75f)/.25f);
    return vec3f(1.f,0.f,0.f);
#endif
  }
  
  OPTIX_RAYGEN_PROGRAM(benchmark)()
  {
    auto &lp = optixLaunchParams;
    const int spp = lp.render.spp; 
    const vec2i threadIdx = owl::getLaunchIndex();
    Ray ray = generateRay(vec2f(threadIdx)+vec2f(.5f));

    vec4f bgColor = vec4f(backGroundColor(),1.f);
    Random random(threadIdx.x,threadIdx.y);

    uint64_t clock_begin = clock();

    vec4f accumColor = 0.f;

    for (int s=0; s<spp; ++s) {
      float rx = random();
      float ry = random();
      float rz = random();

      vec4f color(0.f);
      if (lp.benchmarkMode == BenchmarkModeBinarySearch) {
        Sample sample = sampleCDF(rx,ry);
        color = vec4f((float)sample.x/lp.environmentMapWidth,
                      (float)sample.y/lp.environmentMapHeight,
                      0.f,1.f/spp);
      } else if (lp.benchmarkMode == BenchmarkModeBVH) {
        Sample sample = sampleCDF_BVH(rx,ry);
        color = vec4f((float)sample.x/lp.environmentMapWidth,
                      (float)sample.y/lp.environmentMapHeight,
                      0.f,1.f/spp);
      } else if (lp.benchmarkMode == BenchmarkModeErrors) {
        Sample sample = sampleCDF(rx,ry);
        Sample sampleBVH = sampleCDF_BVH(rx,ry);
        if (sample.x != sampleBVH.x || sample.y != sampleBVH.y) {
          // printf("sample: %i %i %f %f, sampleBVH: %i %i %f %f\n",
          //        sample.x,sample.y,sample.pdfx,sample.pdfy,
          //        sampleBVH.x,sampleBVH.y,sampleBVH.pdfx,sampleBVH.pdfy);
          int off = abs(sample.x-sampleBVH.x)+abs(sample.y-sampleBVH.y);
          off-=1; // as offSamples array is zero-based
          if (off < lp.offSamplesMax-1)
            atomicAdd(&lp.offSamples[off], 1);
          else
            atomicAdd(&lp.offSamples[lp.offSamplesMax-1], 1);
          accumColor=vec4f(1.f);

          vec4f texel = tex2D<float4>(lp.hdrTexture,sample.x,sample.y);
          vec4f texelBVH = tex2D<float4>(lp.hdrTexture,sampleBVH.x,sampleBVH.y);

          // Normalize to min/max luminance range
          texel = (texel-lp.luminanceMin)/(lp.luminanceMax-lp.luminanceMin);
          texelBVH = (texelBVH-lp.luminanceMin)/(lp.luminanceMax-lp.luminanceMin);

          vec3f err(fabsf(texel.x-texelBVH.x),
                    fabsf(texel.y-texelBVH.y),
                    fabsf(texel.y-texelBVH.z));
          float errMax = max(err.x,max(err.y,err.z));
          if (errMax < 1e-30f)
            atomicAdd(&lp.offLuminance[ 0], 1);
          else if (errMax < 1e-20f)
            atomicAdd(&lp.offLuminance[ 1], 1);
          else if (errMax < 1e-15f)
            atomicAdd(&lp.offLuminance[ 2], 1);
          else if (errMax < 1e-10)
            atomicAdd(&lp.offLuminance[ 3], 1);
          else if (errMax < 1e-8)
            atomicAdd(&lp.offLuminance[ 4], 1);
          else if (errMax < 1e-7)
            atomicAdd(&lp.offLuminance[ 5], 1);
          else if (errMax < 1e-6)
            atomicAdd(&lp.offLuminance[ 6], 1);
          else if (errMax < 1e-5)
            atomicAdd(&lp.offLuminance[ 7], 1);
          else if (errMax < 1e-4)
            atomicAdd(&lp.offLuminance[ 8], 1);
          else if (errMax < 1e-3)
            atomicAdd(&lp.offLuminance[ 9], 1);
          else if (errMax < 1e-2)
            atomicAdd(&lp.offLuminance[10], 1);
          else {
            // printf("(%i %i): %f %f %f | (%i %i): %f %f %f\n",
            //        sample.x,sample.y,texel.x,texel.y,texel.z,
            //        sampleBVH.x,sampleBVH.y,texelBVH.x,texelBVH.y,texelBVH.z);
            atomicAdd(&lp.offLuminance[11], 1);
          }
        }
      }
      accumColor = over(color,accumColor);
    }

    uint64_t clock_end = clock();
    if (lp.render.heatMapEnabled > 0.f) {
      float t = (clock_end-clock_begin)*(lp.render.heatMapScale/spp);
      accumColor = over(vec4f(heatMap(t),.5f),accumColor);
    }

    int pixelID = threadIdx.x + owl::getLaunchDims().x*threadIdx.y;
    if (lp.accumID > 0)
      accumColor += vec4f(lp.accumBuffer[pixelID]);
    lp.accumBuffer[pixelID] = accumColor;
    accumColor *= (1.f/(lp.accumID+1));
    
    // bool crossHairs = (owl::getLaunchIndex().x == owl::getLaunchDims().x/2
    //                    ||
    //                    owl::getLaunchIndex().y == owl::getLaunchDims().y/2
    //                    );
    // if (crossHairs) accumColor = vec4f(1.f) - accumColor;
    
    lp.fbPointer[pixelID] = make_rgba(vec3f(accumColor*(1.f/spp)));
  }
  
  OPTIX_RAYGEN_PROGRAM(renderFrame)()
  {
    auto &lp = optixLaunchParams;
    const int spp = lp.render.spp; 
    const vec2i threadIdx = owl::getLaunchIndex();
    int pixelID = threadIdx.x + owl::getLaunchDims().x*threadIdx.y;

    Random random(pixelID,lp.accumID);

    uint64_t clock_begin = clock();

    vec4f accumColor = 0.f;

    for (int s=0; s<spp; ++s) {
      float rx = random();
      float ry = random();
      Ray ray = generateRay(vec2f(threadIdx)+vec2f(rx,ry));
      ModelPRD prd{-1.f,vec3f(-1),-1};
      owl::traceRay(lp.model.group, ray, prd,
                    OPTIX_RAY_FLAG_DISABLE_ANYHIT);
      // face normal forward
      if (dot(ray.direction,prd.gn) > 0.f) {
        prd.gn = -prd.gn;
      }
      vec4f color(0.f);
      if (prd.t_hit >= 0.f) {
        float r1 = random();
        float r2 = random();
        LightSample ls;
        if (lp.renderMode == RenderModeBinarySearch)
          ls = importanceSampleEnvironmentMap(r1,r2);
        else if (lp.renderMode == RenderModeBVH)
          ls = importanceSampleEnvironmentMapBVH(r1,r2);
        else if (lp.renderMode == RenderModeRandom)
          ls = importanceSampleEnvironmentMapRandom(r1,r2);
        else
          assert(0 && "unsupported render mode");
        vec3f isectPos = ray.origin + ray.direction * prd.t_hit;
        ModelPRD shadowPrd = {-1.f,vec3f(-1),-1};
        Ray shadowRay;
        shadowRay.origin = isectPos;
        shadowRay.direction = ls.L;;
        shadowRay.tmin = 1e-2f;
        shadowRay.tmax = 1e20f;
        owl::traceRay(lp.model.group, shadowRay, shadowPrd,
                      OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT);
        ls.intensity = clamp(ls.intensity * lp.hdriIntensity,vec3f(0.f),vec3f(1e30f));
        
        if (shadowPrd.primID < 0)
          color = vec4f(vec3f(.3f + max(0.f,dot(ls.L,prd.gn)))*ls.intensity/ls.pdf,1.f);
        color = min(color, vec4f(100.f)); // clamp out fire flies 
      } else {
        LightSample s = sampleEnvironmentMap(ray.direction);
        color = vec4f(s.intensity,1.f);
      }
      accumColor += color;
    }
    accumColor = accumColor / float(spp);

    uint64_t clock_end = clock();
    if (lp.render.heatMapEnabled > 0.f) {
      float t = (clock_end-clock_begin)*(lp.render.heatMapScale/spp);
      accumColor = over(vec4f(heatMap(t),.5f),accumColor);
    }

    if (lp.accumID > 0)
      accumColor += vec4f(lp.accumBuffer[pixelID]);
    lp.accumBuffer[pixelID] = accumColor;
    accumColor *= (1.f/(lp.accumID+1));
    accumColor.x = linear_to_srgb(accumColor.x);
    accumColor.y = linear_to_srgb(accumColor.y);
    accumColor.z = linear_to_srgb(accumColor.z);
    
    // bool crossHairs = (owl::getLaunchIndex().x == owl::getLaunchDims().x/2
    //                    ||
    //                    owl::getLaunchIndex().y == owl::getLaunchDims().y/2
    //                    );
    // if (crossHairs) accumColor = vec4f(1.f) - accumColor;
    
    lp.fbPointer[pixelID] = make_rgba(vec3f(accumColor));
  }
  
}
