// ======================================================================== //
// Copyright 2019-2020 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#pragma once

#include <owl/owl.h>
#include "owl/common/math/LinearSpace.h"
#include <owl/common/math/vec.h>
#include <cuda_runtime.h>

using namespace owl;

#define RADIANCE_RAY_TYPE 0
#define SHADOW_RAY_TYPE   1

#ifdef __CUDA_ARCH__
typedef owl::RayT<0,2> RadianceRay;
typedef owl::RayT<1,2> ShadowRay;
#endif

struct BasicLight
{
  vec3f pos;
  vec3f color;
  int    casts_shadow; 
};

/* variables for the parallelogram geometry */
struct ParallelogramGeomData
{
  /*! plane */
  vec4f plane;
  /*! other */
  vec3f v1, v2, anchor;
  /* texture objects */
  cudaTextureObject_t ka_map;
  cudaTextureObject_t kd_map;
  cudaTextureObject_t ks_map;
  /*! material */
  struct {
    vec3f Ka;
    vec3f Kd;
    vec3f Ks;
    vec3f reflectivity;
    float phong_exp;
  } material;
};

/* variables for the pool balls geometry */
struct PoolBallsGeomData
{
  /*! array/buffer of center points */
  vec3f *center;
  /*! radius */
  float radius;
  /*! rotations */
  linear3f *rotation;
  /* texture object */
  cudaTextureObject_t *kd_map;
  /*! material */
  struct {
    float importance_cutoff;
    vec3f cutoff_color;
    float fresnel_exponent;
    float fresnel_minimum;
    float fresnel_maximum;
    vec3f reflection_color;
    int reflection_max_depth;
    vec3f  Ka;
    vec3f *Kd;
    vec3f  Ks;
    float  exponent;
  } material;
};

/* variables for the ray generation program */
struct RayGenData
{
  uint32_t *fbPtr;
  vec2i  fbSize;
  OptixTraversableHandle world;

  struct {
    vec3f pos;
    vec3f dir_00;
    vec3f dir_du;
    vec3f dir_dv;
    float focal_scale;
    float aperture_radius;
  } camera;
};

struct LaunchParams
{
  int numLights;
  BasicLight *lights;
  vec3f ambient_light_color;
  float scene_epsilon;
  OptixTraversableHandle world;
  float4   *accumBuffer;
  int       accumID;
};

/* variables for the miss program */
struct MissProgData
{
  vec3f  bg_color;
};

