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
#include "owlViewer/OWLViewer.h"
#include "owl/common/math/AffineSpace.h"
#include <random>

using namespace owl::common;

#define LOG(message)                                            \
  std::cout << OWL_TERMINAL_BLUE;                               \
  std::cout << "#owl.sample(main): " << message << std::endl;   \
  std::cout << OWL_TERMINAL_DEFAULT;
#define LOG_OK(message)                                         \
  std::cout << OWL_TERMINAL_LIGHT_BLUE;                         \
  std::cout << "#owl.sample(main): " << message << std::endl;   \
  std::cout << OWL_TERMINAL_DEFAULT;

extern "C" char ptxCode[];

struct Mesh {
  std::vector<vec3f> vertices;
  std::vector<vec2f> texCoords;
  std::vector<vec3i> indices;
};

// const vec2i fbSize(800,600);
const vec3f init_lookFrom(-4.f,+3.f,-2.f);
const vec3f init_lookAt(0.f,0.f,0.f);
const vec3f init_lookUp(0.f,1.f,0.f);
const float init_cosFovy = 0.66f;

const vec3i numBoxes(4);
const float worldSize = 1;
const vec3f boxSize   = (2*.4f*worldSize)/vec3f(numBoxes);

std::default_random_engine rndGen;
std::uniform_real_distribution<float> distribution_uniform(-1.0f,1.0f);
std::uniform_real_distribution<float> distribution_speed(.1f,.8f);
std::uniform_real_distribution<float> distribution_rot(-.1f,+.1f);
std::uniform_int_distribution<int> distribution_texSize(2,16);

inline vec3f getRandomDir()
{
  vec3f rotationAxis;
  do {
    rotationAxis.x = distribution_uniform(rndGen);
    rotationAxis.y = distribution_uniform(rndGen);
    rotationAxis.z = distribution_uniform(rndGen);
  } while (dot(rotationAxis,rotationAxis) > 1.f);
  return normalize(rotationAxis);
}

void getTransforms(affine3f &xfm0,
                   affine3f &xfm1,
                   vec3i boxID)
{
  const vec3f rotationAxis = getRandomDir();
  
  const float rotationAngle0 = float(distribution_uniform(rndGen)*(2.f*M_PI));
  const linear3f rot0 = linear3f::rotate(rotationAxis,rotationAngle0);
  
  const float rotationAngle1 = float(rotationAngle0+distribution_rot(rndGen));
  const linear3f rot1 = linear3f::rotate(rotationAxis,rotationAngle1);
  
  const vec3f rel = (vec3f(boxID)+.5f) / vec3f(numBoxes);
  const vec3f boxCenter = vec3f(-worldSize) + (2.f*worldSize)*rel;
  const vec3f pos0 = boxCenter;

  const float speed  = distribution_speed(rndGen);
  const vec3f motion = speed * getRandomDir();
  const vec3f pos1 = pos0+motion;;

  xfm0 = affine3f(rot0,pos0);
  xfm1 = affine3f(rot1,pos1);
}

std::vector<affine3f>     boxTransforms0;
std::vector<affine3f>     boxTransforms1;

void addFace(Mesh &mesh, const vec3f ll, const vec3f du, const vec3f dv)
{
  int idxll = (int)mesh.vertices.size();
  for (int iy=0;iy<2;iy++)
    for (int ix=0;ix<2;ix++) {
      mesh.vertices.push_back(ll+float(ix)*du+float(iy)*dv);
      mesh.texCoords.push_back(vec2f((float)ix,(float)iy));
    }
  mesh.indices.push_back(vec3i(idxll,idxll+1,idxll+3));
  mesh.indices.push_back(vec3i(idxll,idxll+3,idxll+2));
}

void addBox(Mesh &mesh, 
            const vec3f du=vec3f(boxSize.x,0,0),
            const vec3f dv=vec3f(0,boxSize.y,0),
            const vec3f dw=vec3f(0,0,boxSize.z))
{
  addFace(mesh,-0.5f*(du+dv+dw),du,dv);
  addFace(mesh,-0.5f*(du+dv+dw),du,dw);
  addFace(mesh,-0.5f*(du+dv+dw),dv,dw);

  addFace(mesh,0.5f*(du+dv+dw),-du,-dv);
  addFace(mesh,0.5f*(du+dv+dw),-du,-dw);
  addFace(mesh,0.5f*(du+dv+dw),-dv,-dw);
}

OWLGroup createBox(OWLContext context,
                 OWLGeomType trianglesGeomType,
                 const vec3i coord)
{
  Mesh mesh;
  addBox(mesh);
  
  // ------------------------------------------------------------------
  // triangle mesh
  // ------------------------------------------------------------------
  OWLBuffer vertexBuffer
    = owlDeviceBufferCreate(context,OWL_FLOAT3,mesh.vertices.size(),mesh.vertices.data());
  OWLBuffer texCoordsBuffer
    = owlDeviceBufferCreate(context,OWL_FLOAT2,mesh.texCoords.size(),mesh.texCoords.data());
  OWLBuffer indexBuffer
    = owlDeviceBufferCreate(context,OWL_INT3,mesh.indices.size(),mesh.indices.data());
  // OWLBuffer frameBuffer
  //   = owlHostPinnedBufferCreate(context,OWL_INT,fbSize.x*fbSize.y);

  OWLGeom trianglesGeom
    = owlGeomCreate(context,trianglesGeomType);

  owlTrianglesSetVertices(trianglesGeom,vertexBuffer,
                          mesh.vertices.size(),sizeof(vec3f),0);
  owlTrianglesSetIndices(trianglesGeom,indexBuffer,
                         mesh.indices.size(),sizeof(vec3i),0);
  
  owlGeomSetBuffer(trianglesGeom,"vertex",vertexBuffer);
  owlGeomSetBuffer(trianglesGeom,"texCoord",texCoordsBuffer);
  owlGeomSetBuffer(trianglesGeom,"index",indexBuffer);
    
  // ------------------------------------------------------------------
  // create a 4x4 checkerboard texture
  // ------------------------------------------------------------------
  vec2i texSize(distribution_texSize(rndGen),distribution_texSize(rndGen));
  vec4uc color0 = vec4uc(255.99f*vec4f((float)distribution_uniform(rndGen),
      (float)distribution_uniform(rndGen),
      (float)distribution_uniform(rndGen),
                                       0.f));
  vec4uc color1 = vec4uc(255)-color0;
  std::vector<vec4uc> texels;
  for (int iy=0;iy<texSize.y;iy++)
    for (int ix=0;ix<texSize.x;ix++) {
      texels.push_back(((ix ^ iy)&1) ?
                       color0 : color1); 
    }
  OWLTexture cbTexture
    = owlTexture2DCreate(context,
                         OWL_TEXEL_FORMAT_RGBA8,
                         texSize.x,texSize.y,
                         texels.data(),
                         OWL_TEXTURE_NEAREST,
                         OWL_TEXTURE_CLAMP);
  owlGeomSetTexture(trianglesGeom,"texture",cbTexture);
  
  // ------------------------------------------------------------------
  // the group/accel for that mesh
  // ------------------------------------------------------------------
  OWLGroup trianglesGroup
    = owlTrianglesGeomGroupCreate(context,1,&trianglesGeom);

  owlGroupBuildAccel(trianglesGroup);

  return trianglesGroup;
}


struct Viewer : public owl::viewer::OWLViewer
{
  Viewer();
  
  /*! gets called whenever the viewer needs us to re-render out widget */
  void render() override;
  
      /*! window notifies us that we got resized. We HAVE to override
          this to know our actual render dimensions, and get pointer
          to the device frame buffer that the viewer cated for us */     
  void resize(const vec2i &newSize) override;

  /*! this function gets called whenever any camera manipulator
    updates the camera. gets called AFTER all values have been updated */
  void cameraChanged() override;

  bool sbtDirty = true;
  OWLRayGen  rayGen  { 0 };
  OWLContext context { 0 };
  OWLGroup   world   { 0 };
};

/*! window notifies us that we got resized */     
void Viewer::resize(const vec2i &newSize)
{
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

  // ----------- set variables  ----------------------------
  owlRayGenSet1ul   (rayGen,"fbPtr",        (uint64_t)fbPointer);
  // owlRayGenSetBuffer(rayGen,"fbPtr",        frameBuffer);
  owlRayGenSet2i    (rayGen,"fbSize",       (const owl2i&)fbSize);
  owlRayGenSet3f    (rayGen,"camera.pos",   (const owl3f&)camera_pos);
  owlRayGenSet3f    (rayGen,"camera.dir_00",(const owl3f&)camera_d00);
  owlRayGenSet3f    (rayGen,"camera.dir_du",(const owl3f&)camera_ddu);
  owlRayGenSet3f    (rayGen,"camera.dir_dv",(const owl3f&)camera_ddv);
  vec3f lightDir = {1.f,1.f,1.f};
  owlRayGenSet3f    (rayGen,"lightDir",     (const owl3f&)lightDir);
  sbtDirty = true;
}

Viewer::Viewer()
{
  // create a context on the first device:
  context = owlContextCreate(nullptr,1);
  owlEnableMotionBlur(context);
  OWLModule module = owlModuleCreate(context,ptxCode);
  
  // ##################################################################
  // set up all the *GEOMETRY* graph we want to render
  // ##################################################################

  // -------------------------------------------------------
  // declare geometry type
  // -------------------------------------------------------
  OWLVarDecl trianglesGeomVars[] = {
    { "index",  OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData,index)},
    { "vertex", OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData,vertex)},
    { "texCoord", OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData,texCoord)},
    { "texture",  OWL_TEXTURE, OWL_OFFSETOF(TrianglesGeomData,texture)},
    { nullptr }
  };
  OWLGeomType trianglesGeomType
    = owlGeomTypeCreate(context,
                        OWL_TRIANGLES,
                        sizeof(TrianglesGeomData),
                        trianglesGeomVars,-1);
  owlGeomTypeSetClosestHit(trianglesGeomType,0,
                           module,"TriangleMesh");

  // ##################################################################
  // set up all the *GEOMS* we want to run that code on
  // ##################################################################

  LOG("building geometries ...");

  // addCube(vec3f(0.f),
  //         vec3f(2.f,0.f,0.f),
  //         vec3f(0.f,2.f,0.f),
  //         vec3f(0.f,0.f,2.f));

  std::vector<OWLGroup> groups;
  for (int iz=0;iz<numBoxes.z;iz++)
    for (int iy=0;iy<numBoxes.y;iy++)
      for (int ix=0;ix<numBoxes.x;ix++) {
        affine3f xfm0, xfm1;
        getTransforms(xfm0,xfm1,vec3i(ix,iy,iz));
        boxTransforms0.push_back(xfm0);
        boxTransforms1.push_back(xfm1);
        groups.push_back(createBox(context,trianglesGeomType,vec3i(ix,iy,iz)));
      }

  world
    = owlInstanceGroupCreate(context,groups.size(),
                             groups.data(),
                             nullptr,
                             nullptr,
                             OWL_MATRIX_FORMAT_OWL);
  // OWLBuffer xfmBuffer0 = owlDeviceBufferCreate(context,OWL_AFFINE3F,
  //                                              boxTransforms0.size(),
  //                                              boxTransforms0.data());
  // OWLBuffer xfmBuffer1 = owlDeviceBufferCreate(context,OWL_AFFINE3F,
  //                                              boxTransforms1.size(),
  //                                              boxTransforms1.data());
  // OWLBuffer xfmArrays[2] = { xfmBuffer0, xfmBuffer1 };
  // owlInstanceGroupSetTransformArrays(world,2,xfmArrays);
  owlInstanceGroupSetTransforms(world,0,(const float*)boxTransforms0.data());
  owlInstanceGroupSetTransforms(world,1,(const float*)boxTransforms1.data());
  owlGroupBuildAccel(world);
  
  // ##################################################################
  // set miss and raygen program required for SBT
  // ##################################################################

  // -------------------------------------------------------
  // set up miss prog 
  // -------------------------------------------------------
  OWLVarDecl missProgVars[]
    = {
    { "color0", OWL_FLOAT3, OWL_OFFSETOF(MissProgData,color0)},
    { "color1", OWL_FLOAT3, OWL_OFFSETOF(MissProgData,color1)},
    { /* sentinel to mark end of list */ }
  };
  // ----------- create object  ----------------------------
  OWLMissProg missProg
    = owlMissProgCreate(context,module,"miss",sizeof(MissProgData),
                        missProgVars,-1);
  
  // ----------- set variables  ----------------------------
  owlMissProgSet3f(missProg,"color0",owl3f{.8f,0.f,0.f});
  owlMissProgSet3f(missProg,"color1",owl3f{.8f,.8f,.8f});

  // -------------------------------------------------------
  // set up ray gen program
  // -------------------------------------------------------
  OWLVarDecl rayGenVars[] = {
    { "fbPtr",         OWL_RAW_POINTER, OWL_OFFSETOF(RayGenData,fbPtr)},
    // { "fbPtr",         OWL_BUFPTR, OWL_OFFSETOF(RayGenData,fbPtr)},
    { "fbSize",        OWL_INT2,   OWL_OFFSETOF(RayGenData,fbSize)},
    { "lightDir",      OWL_FLOAT3, OWL_OFFSETOF(RayGenData,lightDir)},
    { "world",         OWL_GROUP,  OWL_OFFSETOF(RayGenData,world)},
    { "camera.pos",    OWL_FLOAT3, OWL_OFFSETOF(RayGenData,camera.pos)},
    { "camera.dir_00", OWL_FLOAT3, OWL_OFFSETOF(RayGenData,camera.dir_00)},
    { "camera.dir_du", OWL_FLOAT3, OWL_OFFSETOF(RayGenData,camera.dir_du)},
    { "camera.dir_dv", OWL_FLOAT3, OWL_OFFSETOF(RayGenData,camera.dir_dv)},
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
  if (sbtDirty) {
    owlBuildSBT(context);
    sbtDirty = false;
  }
  owlRayGenLaunch2D(rayGen,fbSize.x,fbSize.y);
}


int main(int ac, char **av)
{
  LOG("owl::ng example '" << av[0] << "' starting up");

  Viewer viewer;
  viewer.camera.setOrientation(init_lookFrom,
                               init_lookAt,
                               init_lookUp,
                               owl::viewer::toDegrees(acosf(init_cosFovy)));
  viewer.enableFlyMode();
  viewer.enableInspectMode(owl::box3f(vec3f(-1.f),vec3f(+1.f)));

  // ##################################################################
  // now that everything is ready: launch it ....
  // ##################################################################
  viewer.showAndRun();
}
