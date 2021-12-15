
/*
 * Copyright (c) 2008 - 2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and proprietary
 * rights in and to this software, related documentation and any modifications thereto.
 * Any use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from NVIDIA Corporation is strictly
 * prohibited.
 *
 * TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
 * AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
 * INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
 * SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
 * LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
 * BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
 * INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGES
 */

#pragma once

// ---------------------------------------------------------
// from optixu
// ---------------------------------------------------------

#ifndef M_PIf
#define M_PIf       3.14159265358979323846f
#endif
#ifndef M_PI_2f
#define M_PI_2f     1.57079632679489661923f
#endif
#ifndef M_PI_4f
#define M_PI_4f     0.785398163397448309616f
#endif
#ifndef M_1_PIf
#define M_1_PIf     0.318309886183790671538f
#endif
#ifndef M_2_PIf
#define M_2_PIf     0.636619772367581343076f
#endif

inline __device__ vec3f fminf(const vec3f &a, const vec3f &b)
{
  return vec3f(fminf(a.x,b.x),fminf(a.y,b.y),fminf(a.z,b.z));
}

inline __device__ vec3f fmaxf(const vec3f &a, const vec3f &b)
{
  return vec3f(fmaxf(a.x,b.x),fmaxf(a.y,b.y),fmaxf(a.z,b.z));
}

inline __device__ float fmaxf(const float3& a)
{
  return fmaxf(fmaxf(a.x, a.y), a.z);
}

inline __device__ vec2f square_to_disk(const vec2f& sample)
{
  float phi, r;

  const float a = 2.0f * sample.x - 1.0f;
  const float b = 2.0f * sample.y - 1.0f;

  if (a > -b)
  {
    if (a > b)
    {
      r = a;
      phi = (float)M_PI_4f * (b/a);
    }
    else
    {
      r = b;
      phi = (float)M_PI_4f * (2.0f - (a/b));
    }
  }
  else
  {
    if (a < b)
    {
      r = -a;
      phi = (float)M_PI_4f * (4.0f + (b/a));
    }
    else
    {
      r = -b;
      phi = (b) ? (float)M_PI_4f * (6.0f - (a/b)) : 0.0f;
    }
  }

  return vec2f( r * cosf(phi), r * sinf(phi) );
}

inline __device__ vec3f cart_to_pol(const vec3f& v)
{
  float azimuth;
  float elevation;
  float radius = length(v);

  float r = sqrtf(v.x*v.x + v.y*v.y);
  if (r > 0.0f)
  {
    azimuth   = atanf(v.y / v.x);
    elevation = atanf(v.z / r);

    if (v.x < 0.0f)
      azimuth += M_PIf;
    else if (v.y < 0.0f)
      azimuth += M_PIf * 2.0f;
  }
  else
  {
    azimuth = 0.0f;

    if (v.z > 0.0f)
      elevation = +M_PI_2f;
    else
      elevation = -M_PI_2f;
  }

  return vec3f(azimuth, elevation, radius);
}

inline __device__  float fresnel_schlick(const float cos_theta, const float exponent = 5.0f,
                                         const float minimum = 0.0f, const float maximum = 1.0f)
{
  return clamp(minimum + (maximum - minimum) * powf(fmaxf(0.0f,1.0f - cos_theta), exponent),
               minimum, maximum);
}

inline __device__ vec3f fresnel_schlick(const float cos_theta, const float exponent,
                                        const vec3f& minimum, const vec3f& maximum)
{
  return vec3f(fresnel_schlick(cos_theta, exponent, minimum.x, maximum.x),
               fresnel_schlick(cos_theta, exponent, minimum.y, maximum.y),
               fresnel_schlick(cos_theta, exponent, minimum.z, maximum.z));
}

inline __device__ vec3f reflect(const vec3f& i, const vec3f& n)
{
  return i - 2.0f * n * dot(n,i);
}

inline __device__ vec3f faceforward(const vec3f& n, const vec3f& i, const vec3f& nref)
{
  return n * copysignf( 1.0f, dot(i, nref) );
}

// ---------------------------------------------------------
// 
// ---------------------------------------------------------

// Convert a float3 in [0,1)^3 to a uchar4 in [0,255]^4 -- 4th channel is set to 255
#ifdef __CUDACC__
static __device__ __inline__ uchar4 make_color(const vec3f& c)
{
    return make_uchar4( static_cast<unsigned char>(__saturatef(c.z)*255.99f),  /* B */
                        static_cast<unsigned char>(__saturatef(c.y)*255.99f),  /* G */
                        static_cast<unsigned char>(__saturatef(c.x)*255.99f),  /* R */
                        255u);                                                 /* A */
}
#endif

// Sample Phong lobe relative to U, V, W frame
static
__host__ __device__ __inline__ vec3f sample_phong_lobe( vec2f sample, float exponent, 
                                                        vec3f U, vec3f V, vec3f W )
{
  const float power = expf( logf(sample.y)/(exponent+1.0f) );
  const float phi = sample.x * 2.0f * (float)M_PIf;
  const float scale = sqrtf(1.0f - power*power);
  
  const float x = cosf(phi)*scale;
  const float y = sinf(phi)*scale;
  const float z = power;

  return x*U + y*V + z*W;
}

// Sample Phong lobe relative to U, V, W frame
static
__host__ __device__ __inline__ vec3f sample_phong_lobe( const vec2f &sample, float exponent, 
                                                        const vec3f &U, const vec3f &V, const vec3f &W, 
                                                         float &pdf, float &bdf_val )
{
  const float cos_theta = powf(sample.y, 1.0f/(exponent+1.0f) );

  const float phi = sample.x * 2.0f * M_PIf;
  const float sin_theta = sqrtf(1.0f - cos_theta*cos_theta);
  
  const float x = cosf(phi)*sin_theta;
  const float y = sinf(phi)*sin_theta;
  const float z = cos_theta;

  const float powered_cos = powf( cos_theta, exponent );
  pdf = (exponent+1.0f) / (2.0f*M_PIf) * powered_cos;
  bdf_val = (exponent+2.0f) / (2.0f*M_PIf) * powered_cos;  

  return x*U + y*V + z*W;
}

// Get Phong lobe PDF for local frame
static
__host__ __device__ __inline__ float get_phong_lobe_pdf( float exponent, const vec3f &normal, const vec3f &dir_out, 
                                                         const vec3f &dir_in, float &bdf_val)
{  
  vec3f r = -reflect(dir_out, normal);
  const float cos_theta = fabs(dot(r, dir_in));
  const float powered_cos = powf(cos_theta, exponent );

  bdf_val = (exponent+2.0f) / (2.0f*M_PIf) * powered_cos;  
  return (exponent+1.0f) / (2.0f*M_PIf) * powered_cos;
}

// Create ONB from normal.  Resulting W is parallel to normal
static
__host__ __device__ __inline__ void create_onb( const vec3f& n, vec3f& U, vec3f& V, vec3f& W )
{
  W = normalize( n );
  U = cross( W, vec3f( 0.0f, 1.0f, 0.0f ) );

  if ( fabs( U.x ) < 0.001f && fabs( U.y ) < 0.001f && fabs( U.z ) < 0.001f  )
    U = cross( W, vec3f( 1.0f, 0.0f, 0.0f ) );

  U = normalize( U );
  V = cross( W, U );
}

// Create ONB from normalized vector
static
__device__ __inline__ void create_onb( const vec3f& n, vec3f& U, vec3f& V)
{
  U = cross( n, vec3f( 0.0f, 1.0f, 0.0f ) );

  if ( dot( U, U ) < 1e-3f )
    U = cross( n, vec3f( 1.0f, 0.0f, 0.0f ) );

  U = normalize( U );
  V = cross( n, U );
}

// Compute the origin ray differential for transfer
static
__host__ __device__ __inline__ vec3f differential_transfer_origin(vec3f dPdx, vec3f dDdx, float t, vec3f direction, vec3f normal)
{
  float dtdx = -dot((dPdx + t*dDdx), normal)/dot(direction, normal);
  return (dPdx + t*dDdx)+dtdx*direction;
}

// Compute the direction ray differential for a pinhole camera
static
__host__ __device__ __inline__ vec3f differential_generation_direction(vec3f d, vec3f basis)
{
  float dd = dot(d,d);
  return (dd*basis-dot(d,basis)*d)/(dd*sqrtf(dd));
}

// Compute the direction ray differential for reflection
static
__host__ __device__ __inline__
vec3f differential_reflect_direction(vec3f dPdx, vec3f dDdx, vec3f dNdP, 
                                     vec3f D, vec3f N)
{
  vec3f dNdx = dNdP*dPdx;
  float dDNdx = dot(dDdx,N) + dot(D,dNdx);
  return dDdx - 2.f*(dot(D,N)*dNdx + dDNdx*N);
}

// Compute the direction ray differential for refraction
static __host__ __device__ __inline__ 
vec3f differential_refract_direction(vec3f dPdx, vec3f dDdx, vec3f dNdP, 
                                     vec3f D, vec3f N, float ior, vec3f T)
{
  float eta;
  if(dot(D,N) > 0.f) {
    eta = ior;
    N = -N;
  } else {
    eta = 1.f / ior;
  }

  vec3f dNdx = dNdP*dPdx;
  float mu = eta*dot(D,N)-dot(T,N);
  float TN = -sqrtf(1-eta*eta*(1-dot(D,N)*dot(D,N)));
  float dDNdx = dot(dDdx,N) + dot(D,dNdx);
  float dmudx = (eta - (eta*eta*dot(D,N))/TN)*dDNdx;
  return eta*dDdx - (mu*dNdx+dmudx*N);
}

// Color space conversions
static __host__ __device__ __inline__ vec3f Yxy2XYZ( const vec3f& Yxy )
{
  return vec3f(  Yxy.y * ( Yxy.x / Yxy.z ),
                              Yxy.x,
                              ( 1.0f - Yxy.y - Yxy.z ) * ( Yxy.x / Yxy.z ) );
}

static __host__ __device__ __inline__ vec3f XYZ2rgb( const vec3f& xyz)
{
  const float R = dot( xyz, vec3f(  3.2410f, -1.5374f, -0.4986f ) );
  const float G = dot( xyz, vec3f( -0.9692f,  1.8760f,  0.0416f ) );
  const float B = dot( xyz, vec3f(  0.0556f, -0.2040f,  1.0570f ) );
  return vec3f( R, G, B );
}

static __host__ __device__ __inline__ vec3f Yxy2rgb( vec3f Yxy )
{
  // First convert to xyz
  vec3f xyz = vec3f( Yxy.y * ( Yxy.x / Yxy.z ),
                            Yxy.x,
                            ( 1.0f - Yxy.y - Yxy.z ) * ( Yxy.x / Yxy.z ) );

  const float R = dot( xyz, vec3f(  3.2410f, -1.5374f, -0.4986f ) );
  const float G = dot( xyz, vec3f( -0.9692f,  1.8760f,  0.0416f ) );
  const float B = dot( xyz, vec3f(  0.0556f, -0.2040f,  1.0570f ) );
  return vec3f( R, G, B );
}

static __host__ __device__ __inline__ vec3f rgb2Yxy( vec3f rgb)
{
  // convert to xyz
  const float X = dot( rgb, vec3f( 0.4124f, 0.3576f, 0.1805f ) );
  const float Y = dot( rgb, vec3f( 0.2126f, 0.7152f, 0.0722f ) );
  const float Z = dot( rgb, vec3f( 0.0193f, 0.1192f, 0.9505f ) );

  // convert xyz to Yxy
  return vec3f( Y, 
                      X / ( X + Y + Z ),
                      Y / ( X + Y + Z ) );
}

static __host__ __device__ __inline__ vec3f tonemap( const vec3f &hdr_value, float Y_log_av, float Y_max)
{
  vec3f val_Yxy = rgb2Yxy( hdr_value );
  
  float Y        = val_Yxy.x; // Y channel is luminance
  const float a = 0.04f;
  float Y_rel = a * Y / Y_log_av;
  float mapped_Y = Y_rel * (1.0f + Y_rel / (Y_max * Y_max)) / (1.0f + Y_rel);

  vec3f mapped_Yxy = vec3f( mapped_Y, val_Yxy.y, val_Yxy.z ); 
  vec3f mapped_rgb = Yxy2rgb( mapped_Yxy ); 

  return mapped_rgb;
}

