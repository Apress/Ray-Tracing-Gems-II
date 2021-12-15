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
#include <owl/common/math/random.h>

using namespace owl;

struct Lambertian {
  vec3f albedo;
};

typedef owl::common::LCG<4> Random;

#ifdef __CUDA_ARCH__
inline __device__
float schlick(float cosine,
              float ref_idx)
{
  float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
  r0 = r0 * r0;
  return r0 + (1.0f - r0)*powf((1.0f - cosine), 5.0f);
}

inline __device__
bool refract(const vec3f& v,
             const vec3f& n,
             float ni_over_nt,
             vec3f &refracted)
{
  vec3f uv = normalize(v);
  float dt = dot(uv, n);
  float discriminant = 1.0f - ni_over_nt * ni_over_nt*(1 - dt * dt);
  if (discriminant > 0.f) {
    refracted = ni_over_nt * (uv - n * dt) - n * sqrtf(discriminant);
    return true;
  }
  else
    return false;
}

inline __device__
vec3f reflect(const vec3f &v,
              const vec3f &n)
{
  return v - 2.0f*dot(v, n)*n;
}

typedef enum {
              /*! ray could get properly bounced, and is still alive */
              rayGotBounced,
              /*! ray could not get scattered, and should get cancelled */
              rayGotCancelled,
              /*! ray didn't hit anything, and went into the environemnt */
              rayDidntHitAnything
} ScatterEvent;

/*! "per ray data" (PRD) for our sample's rays. In the simple example, there is only
  one ray type, and it only ever returns one thing, which is a color (everything else
  is handled through the recursion). In addition to that return type, rays have to
  carry recursion state, which in this case are recursion depth and random number state */
struct PerRayData
{
  Random random;
  struct {
    ScatterEvent scatterEvent;
    vec3f        scattered_origin;
    vec3f        scattered_direction;
    vec3f        attenuation;
  } out;
};




inline __device__ vec3f randomPointOnUnitDisc(Random &random) {
  vec3f p;
  do {
    p = 2.0f*vec3f(random(), random(), 0.f) - vec3f(1.f, 1.f, 0.f);
  } while (dot(p, p) >= 1.0f);
  return p;
}


#define RANDVEC3F vec3f(rnd(),rnd(),rnd())

inline __device__ vec3f randomPointInUnitSphere(Random &rnd) {
  vec3f p;
  do {
    p = 2.0f*RANDVEC3F - vec3f(1, 1, 1);
  } while (dot(p,p) >= 1.0f);
  return p;
}


inline __device__
bool scatter(const Lambertian &lambertian,
             const vec3f &P,
             vec3f N,
             // const owl::Ray &ray_in,
             PerRayData &prd)
{
  const vec3f org   = optixGetWorldRayOrigin();
  const vec3f dir   = optixGetWorldRayDirection();

  if (dot(N,dir)  > 0.f)
    N = -N;
  N = normalize(N);

  const vec3f target
    = P + (N + randomPointInUnitSphere(prd.random));


  // return scattering event
  prd.out.scattered_origin    = P;
  prd.out.scattered_direction = (target-P);
  prd.out.attenuation         = lambertian.albedo;
  return true;
}

#endif
