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

// This program sets up a single geometric object, a mesh for a cube, and
// its acceleration structure, then ray traces it.

// public owl node-graph API
#include "owl/owl.h"
// our device-side data structures
#include "deviceCode.h"
// viewer base class, for window and user interaction
#include "owlViewer/InspectMode.h"
#include "owlViewer/OWLViewer.h"
#include "owl/common/math/LinearSpace.h"
#include <random>

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

using namespace owl::common;

#define LOG(message)                                            \
  std::cout << OWL_TERMINAL_BLUE;                               \
  std::cout << "#owl.sample(main): " << message << std::endl;   \
  std::cout << OWL_TERMINAL_DEFAULT;
#define LOG_OK(message)                                         \
  std::cout << OWL_TERMINAL_LIGHT_BLUE;                         \
  std::cout << "#owl.sample(main): " << message << std::endl;   \
  std::cout << OWL_TERMINAL_DEFAULT;

extern "C" char deviceCode_ptx[];

namespace cameraTriangle {
  const vec3f init_lookFrom(0.216657f,-9.58095f,3.37295f);
  const vec3f init_lookAt(3.29424f,-0.882669f,1.06718f);
  const vec3f init_lookUp(0.f,0.f,1.f);
  const float init_cosFovy = 0.66f;
}

namespace camera1984 {
  const vec3f init_lookFrom(3.73523f,4.59409f,12.5919f);
  const vec3f init_lookAt(4.84884f,4.50512f,0.984944f);
  const vec3f init_lookUp(0.174625f,0.984592f,0.00920686f);
  const float init_cosFovy = 0.66f;
}

inline float deg_to_rad( float degrees )
{
  return degrees * (float)M_PI / 180.0f;
}

inline float rad_to_deg( float radians )
{
  return radians * 180.0f / (float)M_PI;
}

inline float random1()
{
  return (float)rand()/(float)RAND_MAX;
}

inline vec2f random2()
{
  return vec2f( random1(), random1() );
}

inline vec3f random3()
{
  return vec3f( random1(), random1(), random1() );
}

inline vec4f random4()
{
  return vec4f( random1(), random1(), random1(), random1() );
}

static linear3f Rotation(vec3f rotation)
{
  float alpha = -deg_to_rad( rotation.x );
  float beta  = -deg_to_rad( rotation.y );
  float gamma = -deg_to_rad( rotation.z );

  float s_a = sinf(alpha);
  float c_a = cosf(alpha);

  float s_b = sinf(beta);
  float c_b = cosf(beta);

  float s_g = sinf(gamma);
  float c_g = cosf(gamma);

  vec3f rotate_x[3] = { {   1,    0,    0   },
                        {   0,   c_a, -s_a  },
                        {   0,   s_a,  c_a  } };

  vec3f rotate_y[3] = { { c_b,   0,   s_b  },
                        {   0,    1,    0  },
                        {-s_b,   0,   c_b  } };

  vec3f rotate_z[3] = { { c_g, -s_g,   0  },
                        { s_g,  c_g,   0  },
                        {   0,    0,    1 } };

  linear3f mat_x(rotate_x[0],rotate_x[1],rotate_x[2]);
  linear3f mat_y(rotate_y[0],rotate_y[1],rotate_y[2]);
  linear3f mat_z(rotate_z[0],rotate_z[1],rotate_z[2]);

  linear3f mat = mat_z * mat_y * mat_x;

  return mat;
}

struct Viewer : public owl::viewer::OWLViewer
{
  enum Setup { SETUP_1984, SETUP_TRIANGLE };

  Viewer(Setup stp = SETUP_TRIANGLE);

  /*! gets called whenever the viewer needs us to re-render out widget */
  void render() override;

      /*! window notifies us that we got resized. We HAVE to override
          this to know our actual render dimensions, and get pointer
          to the device frame buffer that the viewer cated for us */
  void resize(const vec2i &newSize) override;

  /*! this function gets called whenever any camera manipulator
    updates the camera. gets called AFTER all values have been updated */
  void cameraChanged() override;

  Setup setup = SETUP_TRIANGLE;
  // Setup setup = SETUP_1984;

  struct {
    std::vector<vec3f> center;
    std::vector<vec3f> rotation;
    std::vector<vec3f> color;
    std::vector<linear3f> rotationMatrices;
    std::vector<OWLTexture> kd_maps;
    float radius = 1.f;

    OWLBuffer centerBuffer;
    OWLBuffer rotationBuffer;
    OWLBuffer colorBuffer;
    OWLBuffer kd_mapBuffer;
  } poolballs;

  bool sbtDirty = true;
  OWLRayGen  rayGen  { 0 };
  OWLParams  lp      { 0 };
  OWLContext context { 0 };
  OWLGroup   world   { 0 };
  OWLGroup   groups[2]; // [pool balls|parallelogram]
  OWLBuffer  accumBuffer { 0 };
  int accumID { 0 };
};

/*! window notifies us that we got resized */
void Viewer::resize(const vec2i &newSize)
{
  if (!accumBuffer)
    accumBuffer = owlDeviceBufferCreate(context,OWL_FLOAT4,1,nullptr);
  owlBufferResize(accumBuffer,newSize.x*newSize.y);
  owlParamsSetBuffer(lp,"accumBuffer",accumBuffer);
  OWLViewer::resize(newSize);
  cameraChanged();
}

/*! window notifies us that the camera has changed */
void Viewer::cameraChanged()
{
  const vec3f lookFrom = camera.getFrom();
  const vec3f lookAt = camera.getAt();
  const vec3f lookUp = camera.getUp();
  const float cosFovy = camera.getCosFovy();
  // ----------- compute variable values  ------------------
  vec3f camera_pos = lookFrom;
  vec3f camera_d00
    = normalize(lookAt-lookFrom);
  float aspect = fbSize.x / float(fbSize.y);
  vec3f camera_ddu
    = cosFovy * aspect * normalize(cross(camera_d00,lookUp));
  vec3f camera_ddv
    = cosFovy * normalize(cross(camera_ddu,camera_d00));
  camera_d00 -= 0.5f * camera_ddu;
  camera_d00 -= 0.5f * camera_ddv;
  accumID = 0;

  float focal_distance = length(lookAt-lookFrom);// + (-1.5f); // 0.f for 1984 scene!
  focal_distance = fmaxf(focal_distance, 1e-2f);
  //float focal_scale = focal_distance / length(lookAt-lookFrom);std::cout << focal_scale << '\n';
  float focal_scale = 10.f;//focal_distance;

  // ----------- set variables  ----------------------------
  owlRayGenSet1ul   (rayGen,"fbPtr",        (uint64_t)fbPointer);
  // owlRayGenSetBuffer(rayGen,"fbPtr",        frameBuffer);
  owlRayGenSet2i    (rayGen,"fbSize",       (const owl2i&)fbSize);
  owlRayGenSet3f    (rayGen,"camera.pos",   (const owl3f&)camera_pos);
  owlRayGenSet3f    (rayGen,"camera.dir_00",(const owl3f&)camera_d00);
  owlRayGenSet3f    (rayGen,"camera.dir_du",(const owl3f&)camera_ddu);
  owlRayGenSet3f    (rayGen,"camera.dir_dv",(const owl3f&)camera_ddv);
  // DoF camera setup
  owlRayGenSet1f    (rayGen,"camera.aperture_radius",setup==SETUP_TRIANGLE ? .15f : .1f);
  owlRayGenSet1f    (rayGen,"camera.focal_scale",focal_scale);

  sbtDirty = true;
}

Viewer::Viewer(Viewer::Setup stp)
  : setup(stp)
{
  // create a context on the first device:
  context = owlContextCreate(nullptr,1);
  owlContextSetRayTypeCount(context,2);
  OWLModule module = owlModuleCreate(context,deviceCode_ptx);

  // ##################################################################
  // set up all the *GEOMETRY* graph we want to render
  // ##################################################################

  // -------------------------------------------------------
  // declare geometry types
  // -------------------------------------------------------
  OWLVarDecl poolBallsGeomVars[] = {
    { "center", OWL_BUFPTR, OWL_OFFSETOF(PoolBallsGeomData,center)},
    { "radius", OWL_FLOAT, OWL_OFFSETOF(PoolBallsGeomData,radius)},
    { "rotation", OWL_BUFPTR, OWL_OFFSETOF(PoolBallsGeomData,rotation)},
    { "kd_map",  OWL_BUFPTR, OWL_OFFSETOF(PoolBallsGeomData,kd_map)},
    { "material.importance_cutoff", OWL_FLOAT, OWL_OFFSETOF(PoolBallsGeomData,material.importance_cutoff)},
    { "material.cutoff_color", OWL_FLOAT3, OWL_OFFSETOF(PoolBallsGeomData,material.cutoff_color)},
    { "material.fresnel_exponent", OWL_FLOAT, OWL_OFFSETOF(PoolBallsGeomData,material.fresnel_exponent)},
    { "material.fresnel_minimum", OWL_FLOAT, OWL_OFFSETOF(PoolBallsGeomData,material.fresnel_minimum)},
    { "material.fresnel_maximum", OWL_FLOAT, OWL_OFFSETOF(PoolBallsGeomData,material.fresnel_maximum)},
    { "material.reflection_color", OWL_FLOAT3, OWL_OFFSETOF(PoolBallsGeomData,material.reflection_color)},
    { "material.reflection_max_depth", OWL_INT, OWL_OFFSETOF(PoolBallsGeomData,material.reflection_max_depth)},
    { "material.Ka", OWL_FLOAT3, OWL_OFFSETOF(PoolBallsGeomData,material.Ka)},
    { "material.Kd", OWL_BUFPTR, OWL_OFFSETOF(PoolBallsGeomData,material.Kd)},
    { "material.Ks", OWL_FLOAT3, OWL_OFFSETOF(PoolBallsGeomData,material.Ks)},
    { "material.exponent", OWL_FLOAT, OWL_OFFSETOF(PoolBallsGeomData,material.exponent)},
    { nullptr }
  };
  OWLGeomType poolBallsGeomType
    = owlGeomTypeCreate(context,
                        OWL_GEOMETRY_USER,
                        sizeof(PoolBallsGeomData),
                        poolBallsGeomVars,-1);
  owlGeomTypeSetBoundsProg(poolBallsGeomType,
                           module,"PoolBall");
  owlGeomTypeSetClosestHit(poolBallsGeomType,RADIANCE_RAY_TYPE,
                           module,"PoolBall");
  owlGeomTypeSetIntersectProg(poolBallsGeomType,RADIANCE_RAY_TYPE,
                              module,"PoolBall");
  owlGeomTypeSetAnyHit(poolBallsGeomType,SHADOW_RAY_TYPE,
                       module,"PoolBall");
  owlGeomTypeSetIntersectProg(poolBallsGeomType,SHADOW_RAY_TYPE,
                              module,"PoolBall");

  OWLVarDecl parallelogramGeomVars[] = {
    { "plane", OWL_FLOAT4, OWL_OFFSETOF(ParallelogramGeomData,plane)},
    { "v1", OWL_FLOAT3, OWL_OFFSETOF(ParallelogramGeomData,v2)},
    { "v2", OWL_FLOAT3, OWL_OFFSETOF(ParallelogramGeomData,v1)},
    { "anchor", OWL_FLOAT3, OWL_OFFSETOF(ParallelogramGeomData,anchor)},
    { "ka_map",  OWL_TEXTURE, OWL_OFFSETOF(ParallelogramGeomData,ka_map)},
    { "kd_map",  OWL_TEXTURE, OWL_OFFSETOF(ParallelogramGeomData,kd_map)},
    { "ks_map",  OWL_TEXTURE, OWL_OFFSETOF(ParallelogramGeomData,ks_map)},
    { "material.Ka", OWL_FLOAT3, OWL_OFFSETOF(ParallelogramGeomData,material.Ka)},
    { "material.Kd", OWL_FLOAT3, OWL_OFFSETOF(ParallelogramGeomData,material.Kd)},
    { "material.Ks", OWL_FLOAT3, OWL_OFFSETOF(ParallelogramGeomData,material.Ks)},
    { "material.reflectivity", OWL_FLOAT3, OWL_OFFSETOF(ParallelogramGeomData,material.reflectivity)},
    { "material.phong_exp", OWL_FLOAT, OWL_OFFSETOF(ParallelogramGeomData,material.phong_exp)},
    { nullptr }
  };
  OWLGeomType parallelogramGeomType
    = owlGeomTypeCreate(context,
                        OWL_GEOMETRY_USER,
                        sizeof(ParallelogramGeomData),
                        parallelogramGeomVars,-1);
  owlGeomTypeSetBoundsProg(parallelogramGeomType,
                           module,"Parallelogram");
  owlGeomTypeSetClosestHit(parallelogramGeomType,RADIANCE_RAY_TYPE,
                           module,"Parallelogram");
  owlGeomTypeSetIntersectProg(parallelogramGeomType,RADIANCE_RAY_TYPE,
                              module,"Parallelogram");
  owlGeomTypeSetAnyHit(parallelogramGeomType,SHADOW_RAY_TYPE,
                       module,"Parallelogram");
  owlGeomTypeSetIntersectProg(parallelogramGeomType,SHADOW_RAY_TYPE,
                              module,"Parallelogram");

  // Call this so we have the bounds progs available
  owlBuildPrograms(context);

  // ##################################################################
  // set up all the *GEOMS* we want to run that code on
  // ##################################################################

  LOG("building geometries ...");

  if (setup==SETUP_TRIANGLE) {

    poolballs.center.resize(16);
    poolballs.rotation.resize(16);
    poolballs.color.resize(16);
    poolballs.rotationMatrices.resize(16);
    poolballs.kd_maps.resize(16);

    const char *ppmFileNames[] = {
      "pool_1.ppm",
      "pool_2.ppm",
      "pool_14.ppm",
      "pool_11.ppm",
      "pool_8.ppm",
      "pool_5.ppm",
      "pool_3.ppm",
      "pool_4.ppm",
      "pool_9.ppm",
      "pool_13.ppm",
      "pool_12.ppm",
      "pool_6.ppm",
      "pool_15.ppm",
      "pool_10.ppm",
      "pool_7.ppm",
    };

    unsigned index = 0;
    for (unsigned i=0; i<5; i++)
      for (unsigned j=0; j<=i; j++)
      {
        // pool triangle
        poolballs.center[index].x = 2.f*(float)j-(float)i;
        poolballs.center[index].y = 2.f*(float)i;
        poolballs.center[index].z = 1.f;

         // scale and translate
        poolballs.center[index] *= 1.f;
        poolballs.center[index] += vec3f(5.f, .5f, 0.f);

        // random rotation
        poolballs.rotation[index] = 90.f * ( 2.f * random3() - vec3f(1.f, 1.f, 1.f) );
        poolballs.rotationMatrices[index] = Rotation(poolballs.rotation[index]);

        poolballs.color[index] = vec3f(1.f,.96f,.94f);

        vec2i res;
        int  comp;
        std::string path(DATA_PATH);
        path += "/"+std::string(ppmFileNames[index]);
        unsigned char* image = stbi_load(path.c_str(),
                                         &res.x, &res.y, &comp, STBI_rgb);

        // oof, rather implement OWL_RGB8.. :-)
        std::vector<unsigned char> texels(res.x*res.y*4);
        for (int y=res.y-1; y>=0; --y) {
          for (int x=0; x<res.x; ++x) {
            int index = (y*res.x+x)*4;
            texels[index]=*image++;
            texels[index+1]=*image++;
            texels[index+2]=*image++;
            texels[index+3]=(comp==3) ? 1U : *image++;
          }
        }
        poolballs.kd_maps[index]
          = owlTexture2DCreate(context,
                               OWL_TEXEL_FORMAT_RGBA8,
                               res.x,res.y,
                               texels.data(),
                               OWL_TEXTURE_NEAREST,
                               OWL_TEXTURE_CLAMP);
        index++;
      }

    //Nova
    poolballs.center[15] = { 3.f, -20.f, 1.f };
    poolballs.rotation[15] = poolballs.rotation[14];
    poolballs.rotationMatrices[15] = poolballs.rotationMatrices[14];
    poolballs.color[15] = poolballs.color[14];
  } else {
    poolballs.center.resize(5);
    poolballs.rotation.resize(5);
    poolballs.color.resize(5);
    poolballs.rotationMatrices.resize(5);
    poolballs.kd_maps.resize(5);

    poolballs.center[0] = { 1.85f, 6.28f, 1.f };
    poolballs.rotation[0] = { -0.872f-90.f, 28.322f, -14.422f };
    poolballs.rotationMatrices[0] = Rotation(poolballs.rotation[0]);
    poolballs.color[0] = { 1.f, .96f, .94f };

    poolballs.center[1] = { 4.37f, 5.19f, 1.f };
    poolballs.rotation[1] = { 11.339f-90.f, 14.126f, -17.257f };
    poolballs.rotationMatrices[1] = Rotation(poolballs.rotation[1]);
    poolballs.color[1] = { 1.f, .96f, .94f };

    poolballs.center[2] = { 6.23f, 6.07f, 1.f };
    poolballs.rotation[2] = { 15.492f-90.f, 16.372f, -21.111f };
    poolballs.rotationMatrices[2] = Rotation(poolballs.rotation[2]);
    poolballs.color[2] = { 1.f, .96f, .94f };

    poolballs.center[3] = { 8.31f, 6.83f, 1.f };
    poolballs.rotation[3] = { 27.679f-90.f, -2.611f, -16.905f };
    poolballs.rotationMatrices[3] = Rotation(poolballs.rotation[3]);
    poolballs.color[3] = { 1.f, .96f, .94f };

    poolballs.center[4] = { 3.93f, 1.91f, 1.f };
    poolballs.rotation[4] = { 0.f, 0.f, 0.f };
    poolballs.rotationMatrices[4] = Rotation(poolballs.rotation[4]);
    poolballs.color[4] = { 1.f, .96f, .94f };

    const char *ppmFileNames[] = {
      "pool_1.ppm",
      "pool_9.ppm",
      "pool_8.ppm",
      "pool_4.ppm",
    };

    for (int i=0; i<4; ++i) {
      vec2i res;
      int  comp;
      std::string path(DATA_PATH);
      path += "/"+std::string(ppmFileNames[i]);
      unsigned char* image = stbi_load(path.c_str(),
                                       &res.x, &res.y, &comp, STBI_rgb);

      // oof, rather implement OWL_RGB8.. :-)
      std::vector<unsigned char> texels(res.x*res.y*4);
      for (int y=res.y-1; y>=0; --y) {
        for (int x=0; x<res.x; ++x) {
          int index = (y*res.x+x)*4;
          texels[index]=*image++;
          texels[index+1]=*image++;
          texels[index+2]=*image++;
          texels[index+3]=(comp==3) ? 1U : *image++;
        }
      }
      poolballs.kd_maps[i]
        = owlTexture2DCreate(context,
                             OWL_TEXEL_FORMAT_RGBA8,
                             res.x,res.y,
                             texels.data(),
                             OWL_TEXTURE_NEAREST,
                             OWL_TEXTURE_CLAMP);
    }
  }

  poolballs.centerBuffer
    = owlDeviceBufferCreate(context,OWL_FLOAT3,poolballs.center.size(),&poolballs.center[0]);
  poolballs.rotationBuffer
    = owlDeviceBufferCreate(context,OWL_USER_TYPE(linear3f),poolballs.rotationMatrices.size(),&poolballs.rotationMatrices[0]);
  poolballs.colorBuffer
    = owlDeviceBufferCreate(context,OWL_FLOAT3,poolballs.color.size(),&poolballs.color[0]);
  poolballs.kd_mapBuffer
    = owlDeviceBufferCreate(context,OWL_TEXTURE,poolballs.kd_maps.size(),&poolballs.kd_maps[0]);

  OWLGeom poolBallsGeom = owlGeomCreate(context,poolBallsGeomType);
  owlGeomSetPrimCount(poolBallsGeom,poolballs.center.size());
  owlGeomSetBuffer(poolBallsGeom,"center",poolballs.centerBuffer);
  owlGeomSet1f(poolBallsGeom,"radius",poolballs.radius);
  owlGeomSetBuffer(poolBallsGeom,"rotation",poolballs.rotationBuffer);
  owlGeomSetBuffer(poolBallsGeom,"material.Kd",poolballs.colorBuffer);
  owlGeomSetBuffer(poolBallsGeom,"kd_map",poolballs.kd_mapBuffer);

  owlGeomSet1f(poolBallsGeom,"material.importance_cutoff",1e-2f);
  owlGeomSet3f(poolBallsGeom,"material.cutoff_color",.34f,.55f,.85f);
  owlGeomSet1f(poolBallsGeom,"material.fresnel_exponent",4.f);
  owlGeomSet1f(poolBallsGeom,"material.fresnel_minimum",.1f);
  owlGeomSet1f(poolBallsGeom,"material.fresnel_maximum",1.f);
  owlGeomSet3f(poolBallsGeom,"material.reflection_color",1.f,1.f,1.f);
  owlGeomSet1i(poolBallsGeom,"material.reflection_max_depth",5);
  owlGeomSet3f(poolBallsGeom,"material.Ka",.2f,.2f,.2f);
  owlGeomSet3f(poolBallsGeom,"material.Ks",1.f,1.f,1.f);
  owlGeomSet1f(poolBallsGeom,"material.exponent",1024.f);

  groups[0] = owlUserGeomGroupCreate(context,1,&poolBallsGeom);
  owlGroupBuildAccel(groups[0]);


  vec3f anchor( -5.00f, -25.0f, 0.01f );
  vec3f v1( 20.0f,  0.0f, 0.01f );
  vec3f v2(  0.0f, 50.0f, 0.01f );

  vec3f normal = cross( v1, v2 );
  normal = normalize( normal );
  float d = dot( normal, anchor );
  v1 *= 1.0f/dot( v1, v1 );
  v2 *= 1.0f/dot( v2, v2 );
  vec4f plane( normal, d );

  vec2i res;
  int comp;
  std::string path(DATA_PATH);
  path += "/cloth.ppm";
  unsigned char* image = stbi_load(path.c_str(),
                                   &res.x, &res.y, &comp, STBI_rgb);
  // oof, rather implement OWL_RGB8.. :-)
  std::vector<unsigned char> texels(res.x*res.y*4);
  for (int y=res.y-1; y>=0; --y) {
    for (int x=0; x<res.x; ++x) {
      int index = (y*res.x+x)*4;
      texels[index]=*image++;
      texels[index+1]=*image++;
      texels[index+2]=*image++;
      texels[index+3]=(comp==3) ? 1U : *image++;
    }
  }
  OWLTexture ka_map
    = owlTexture2DCreate(context,
                         OWL_TEXEL_FORMAT_RGBA8,
                         res.x,res.y,
                         texels.data(),
                         OWL_TEXTURE_NEAREST,
                         OWL_TEXTURE_CLAMP);
  OWLTexture kd_map
    = owlTexture2DCreate(context,
                         OWL_TEXEL_FORMAT_RGBA8,
                         res.x,res.y,
                         texels.data(),
                         OWL_TEXTURE_NEAREST,
                         OWL_TEXTURE_CLAMP);

  OWLGeom parallelogramGeom = owlGeomCreate(context,parallelogramGeomType);
  owlGeomSetPrimCount(parallelogramGeom,1);
  owlGeomSet4f(parallelogramGeom,"plane",plane.x,plane.y,plane.z,plane.w);
  owlGeomSet3f(parallelogramGeom,"v1",v1.x,v1.y,v1.z);
  owlGeomSet3f(parallelogramGeom,"v2",v2.x,v2.y,v2.z);
  owlGeomSet3f(parallelogramGeom,"anchor",anchor.x,anchor.y,anchor.z);
  owlGeomSetTexture(parallelogramGeom,"ka_map",ka_map);
  owlGeomSetTexture(parallelogramGeom,"kd_map",kd_map);
  owlGeomSet3f(parallelogramGeom,"material.Ka",.35f,.35f,.35f);
  owlGeomSet3f(parallelogramGeom,"material.Kd",.5f,.5f,.5f);
  owlGeomSet3f(parallelogramGeom,"material.Ks",1.f,1.f,1.f);
  owlGeomSet3f(parallelogramGeom,"material.reflectivity",0.f,0.f,0.f);
  owlGeomSet1f(parallelogramGeom,"material.phong_exp",1.f);

  groups[1] = owlUserGeomGroupCreate(context,1,&parallelogramGeom);
  owlGroupBuildAccel(groups[1]);
  world
    = owlInstanceGroupCreate(context,2,
                             groups,
                             nullptr,
                             nullptr,
                             OWL_MATRIX_FORMAT_OWL);
  owlGroupBuildAccel(world);

  // ##################################################################
  // set miss and raygen program required for SBT
  // ##################################################################

  // -------------------------------------------------------
  // set up miss prog
  // -------------------------------------------------------
  OWLVarDecl missProgVars[]
    = {
    { "bg_color", OWL_FLOAT3, OWL_OFFSETOF(MissProgData,bg_color)},
    { /* sentinel to mark end of list */ }
  };
  // ----------- create object  ----------------------------
  OWLMissProg missProg
    = owlMissProgCreate(context,module,"miss",sizeof(MissProgData),
                        missProgVars,-1);

  // ----------- set variables  ----------------------------
  owlMissProgSet3f(missProg,"bg_color",owl3f{.34f,.55f,.85f});

  // -------------------------------------------------------
  // set up launch params
  // -------------------------------------------------------
  OWLVarDecl launchParamsVars[] = {
    { "world",         OWL_GROUP,  OWL_OFFSETOF(LaunchParams,world)},
    { "accumBuffer",   OWL_BUFPTR, OWL_OFFSETOF(LaunchParams,accumBuffer) },
    { "accumID",   OWL_INT, OWL_OFFSETOF(LaunchParams,accumID) },
    { "numLights",     OWL_INT, OWL_OFFSETOF(LaunchParams,numLights)},
    { "lights",        OWL_BUFPTR, OWL_OFFSETOF(LaunchParams,lights)},
    { "ambient_light_color", OWL_FLOAT3, OWL_OFFSETOF(LaunchParams,ambient_light_color)},
    { "scene_epsilon", OWL_FLOAT, OWL_OFFSETOF(LaunchParams,scene_epsilon)},
    { nullptr/* sentinel to mark end of list */ }
  };

  // ----------- create object  ----------------------------
  lp = owlParamsCreate(context,sizeof(LaunchParams),launchParamsVars,-1);

  owlParamsSetGroup (lp,"world",        world);

  /* light sources */
  owlParamsSet1i(lp,"numLights",2);
  BasicLight lights[] = {
    { vec3f( -30.0f, -10.0f, 80.0f ), vec3f( 1.0f, 1.0f, 1.0f ), 1 },
    { vec3f(  10.0f,  30.0f, 20.0f ), vec3f( 1.0f, 1.0f, 1.0f ), 1 }
  };
  OWLBuffer lightBuffer
    = owlDeviceBufferCreate(context,OWL_USER_TYPE(BasicLight),2,&lights);
  owlParamsSetBuffer(lp,"lights",lightBuffer);
  owlParamsSet3f(lp,"ambient_light_color",.4f,.4f,.4f);
  owlParamsSet1f(lp,"scene_epsilon",1e-2f);

  // -------------------------------------------------------
  // set up ray gen program
  // -------------------------------------------------------
  OWLVarDecl rayGenVars[] = {
    { "fbPtr",         OWL_RAW_POINTER, OWL_OFFSETOF(RayGenData,fbPtr)},
    // { "fbPtr",         OWL_BUFPTR, OWL_OFFSETOF(RayGenData,fbPtr)},
    { "fbSize",        OWL_INT2,   OWL_OFFSETOF(RayGenData,fbSize)},
    { "world",         OWL_GROUP,  OWL_OFFSETOF(RayGenData,world)},
    { "camera.pos",    OWL_FLOAT3, OWL_OFFSETOF(RayGenData,camera.pos)},
    { "camera.dir_00", OWL_FLOAT3, OWL_OFFSETOF(RayGenData,camera.dir_00)},
    { "camera.dir_du", OWL_FLOAT3, OWL_OFFSETOF(RayGenData,camera.dir_du)},
    { "camera.dir_dv", OWL_FLOAT3, OWL_OFFSETOF(RayGenData,camera.dir_dv)},
    { "camera.aperture_radius",    OWL_FLOAT, OWL_OFFSETOF(RayGenData,camera.aperture_radius)},
    { "camera.focal_scale",    OWL_FLOAT, OWL_OFFSETOF(RayGenData,camera.focal_scale)},
    { nullptr/* sentinel to mark end of list */ }
  };

  // ----------- create object  ----------------------------
  rayGen
    = owlRayGenCreate(context,module,"simpleRayGen",
                      sizeof(RayGenData),
                      rayGenVars,-1);

  /* camera and frame buffer get set in resiez() and cameraChanged() */
  owlRayGenSetGroup (rayGen,"world",        world);

  // ##################################################################
  // build *SBT* required to trace the groups
  // ##################################################################

  owlBuildPrograms(context);
  owlBuildPipeline(context);
  owlBuildSBT(context);
  sbtDirty = true;
}

void Viewer::render()
{
  if (setup == SETUP_1984) {
  	// Jitter the location of the pool ball for motion blur
  	vec3f offset = random1() * vec3f(0.1f, 0.6f, 0.0f);
    std::vector<vec3f> center(poolballs.center);
    center[4] += offset;
    owlBufferUpload(poolballs.centerBuffer,center.data());
    owlGroupBuildAccel(groups[0]);
    owlGroupBuildAccel(world);
  }

  if (sbtDirty) {
    owlBuildSBT(context);
    sbtDirty = false;
  }
  owlParamsSet1i(lp,"accumID",accumID);
  accumID++;
  owlLaunch2D(rayGen,fbSize.x,fbSize.y,lp);
}


int main(int ac, char **av)
{
  LOG("owl::ng example '" << av[0] << "' starting up");

  std::string arg1;
  if (ac>1) {
    arg1 = std::string(av[1]);
    if (arg1=="-h") {
      std::cout << "Usage: " << av[0] << "[-h|-1984]\n";
      std::cout << "  -h:    print this message\n";
      std::cout << "  -1984: load the 1984 motion blur scene\n";
      exit(EXIT_SUCCESS);
    }
  }

  Viewer viewer(arg1=="-1984"? Viewer::SETUP_1984: Viewer::SETUP_TRIANGLE);
  if (viewer.setup==Viewer::SETUP_TRIANGLE) {
    viewer.camera.setOrientation(cameraTriangle::init_lookFrom,
                                 cameraTriangle::init_lookAt,
                                 cameraTriangle::init_lookUp,
                                 owl::viewer::toDegrees(acosf(cameraTriangle::init_cosFovy)));
  } else {
    viewer.camera.setOrientation(camera1984::init_lookFrom,
                                 camera1984::init_lookAt,
                                 camera1984::init_lookUp,
                                 owl::viewer::toDegrees(acosf(camera1984::init_cosFovy)));
  }
  viewer.enableFlyMode();
  viewer.enableInspectMode(viewer::OWLViewer::Arcball,
                           owl::box3f(vec3f(-10.f),vec3f(+10.f)));

  // ##################################################################
  // now that everything is ready: launch it ....
  // ##################################################################
  viewer.showAndRun();
}
