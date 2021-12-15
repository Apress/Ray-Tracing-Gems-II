// ======================================================================== //
// Copyright 2018-2021 Ingo Wald                                            //
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

#include "SampleRenderer.h"
#include "LaunchParams.h"
#include <string.h>
#include <fstream>

/*! \namespace osc - Optix Siggraph Course */
namespace osc {

  extern "C" char devicePrograms_ptx[];

  /*! constructor - performs all setup, including initializing
    optix, creates module, pipeline, programs, SBT, etc. */
  SampleRenderer::SampleRenderer(const Model *model, const QuadLight &light)
    : model(model)
  {
    std::cout << "for now, create exactly one device" << std::endl;
    context = owlContextCreate(nullptr,1);
    owlContextSetRayTypeCount(context,2);

    module = owlModuleCreate(context,devicePrograms_ptx);
    rayGen
      = owlRayGenCreate(context,module,"renderFrame",
                        /* no sbt data: */0,nullptr,-1);
    missProgRadiance
      = owlMissProgCreate(context,module,"radiance",
                          /* no sbt data: */0,nullptr,-1);
    missProgShadow
      = owlMissProgCreate(context,module,"shadow",
                          /* no sbt data: */0,nullptr,-1);

    OWLVarDecl launchParamsVars[] = {
      { "world", OWL_GROUP, OWL_OFFSETOF(LaunchParams,traversable)},

      { "numPixelSamples", OWL_INT,    OWL_OFFSETOF(LaunchParams,numPixelSamples)},

      { "frame.frameID", OWL_INT,    OWL_OFFSETOF(LaunchParams,frame.frameID)},
      { "frame.fbColor",OWL_BUFPTR,OWL_OFFSETOF(LaunchParams,frame.fbColor)},
      { "frame.fbFinal",OWL_RAW_POINTER,OWL_OFFSETOF(LaunchParams,frame.fbFinal)},
      { "frame.fbSize",OWL_INT2,OWL_OFFSETOF(LaunchParams,frame.fbSize)},

      // light settings:
      { "light.origin",    OWL_FLOAT3, OWL_OFFSETOF(LaunchParams,light.origin)},
      { "light.du",    OWL_FLOAT3, OWL_OFFSETOF(LaunchParams,light.du)},
      { "light.dv",    OWL_FLOAT3, OWL_OFFSETOF(LaunchParams,light.dv)},
      { "light.power",    OWL_FLOAT3, OWL_OFFSETOF(LaunchParams,light.power)},

      // camera settings:
      { "camera.position", OWL_FLOAT3, OWL_OFFSETOF(LaunchParams,camera.position)},
      { "camera.direction", OWL_FLOAT3, OWL_OFFSETOF(LaunchParams,camera.direction)},
      { "camera.horizontal", OWL_FLOAT3, OWL_OFFSETOF(LaunchParams,camera.horizontal)},
      { "camera.vertical", OWL_FLOAT3, OWL_OFFSETOF(LaunchParams,camera.vertical)},
      { nullptr /* sentinel to mark end of list */ }
    };
    launchParams
      = owlParamsCreate(context,sizeof(LaunchParams),
                              launchParamsVars,-1);

    createTextures();
    buildAccel();

    owlBuildPrograms(context);
    owlBuildPipeline(context);
    owlBuildSBT(context);

    owlParamsSet3f(launchParams,"light.origin",(const owl3f&)light.origin);
    owlParamsSet3f(launchParams,"light.du",    (const owl3f&)light.du);
    owlParamsSet3f(launchParams,"light.dv",    (const owl3f&)light.dv);
    owlParamsSet3f(launchParams,"light.power", (const owl3f&)light.power);
  }

  void SampleRenderer::createTextures()
  {
    int numTextures = (int)model->textures.size();

    textures.resize(numTextures);

    for (int textureID=0;textureID<numTextures;textureID++) {
      auto texture = model->textures[textureID];

      int32_t width  = texture->resolution.x;
      int32_t height = texture->resolution.y;
      this->textures[textureID]
        = owlTexture2DCreate(context,
                             OWL_TEXEL_FORMAT_RGBA8,
                             width,
                             height,
                             texture->pixel,
                             OWL_TEXTURE_LINEAR,
                             OWL_TEXTURE_CLAMP);
    }
  }

  void SampleRenderer::buildAccel()
  {
    const int numMeshes = (int)model->meshes.size();
    std::vector<OWLGeom> meshes;

    OWLVarDecl triMeshVars[] = {
      { "color",    OWL_FLOAT3, OWL_OFFSETOF(TriangleMeshSBTData,color) },
      { "vertex",   OWL_BUFPTR, OWL_OFFSETOF(TriangleMeshSBTData,vertex) },
      { "normal",   OWL_BUFPTR, OWL_OFFSETOF(TriangleMeshSBTData,normal) },
      { "index",    OWL_BUFPTR, OWL_OFFSETOF(TriangleMeshSBTData,index) },
      { "texcoord", OWL_BUFPTR, OWL_OFFSETOF(TriangleMeshSBTData,texcoord) },
      { "hasTexture",OWL_INT,
        OWL_OFFSETOF(TriangleMeshSBTData,hasTexture) },
      { "texture",  OWL_TEXTURE,// OWL_USER_TYPE(cudaTextureObject_t),
        OWL_OFFSETOF(TriangleMeshSBTData,texture) },
      { nullptr /* sentinel to mark end of list */ }
    };
    OWLGeomType triMeshGeomType
      = owlGeomTypeCreate(context,
                          OWL_GEOM_TRIANGLES,
                          sizeof(TriangleMeshSBTData),
                          triMeshVars,-1);
    owlGeomTypeSetClosestHit(triMeshGeomType,0,module,"radiance");
    owlGeomTypeSetClosestHit(triMeshGeomType,1,module,"shadow");
    std::vector<OWLGeom> geoms;
    for (int meshID=0;meshID<numMeshes;meshID++) {
      // upload the model to the device: the builder
      TriangleMesh &mesh = *model->meshes[meshID];

      OWLBuffer vertexBuffer
        = owlDeviceBufferCreate(context,OWL_FLOAT3,mesh.vertex.size(),
                                mesh.vertex.data());
      OWLBuffer indexBuffer
        = owlDeviceBufferCreate(context,OWL_INT3,mesh.index.size(),
                                mesh.index.data());
      OWLBuffer normalBuffer
        = mesh.normal.empty()
        ? nullptr
        : owlDeviceBufferCreate(context,OWL_FLOAT3,mesh.normal.size(),
                                mesh.normal.data());
      OWLBuffer texcoordBuffer
        = mesh.texcoord.empty()
        ? nullptr
        : owlDeviceBufferCreate(context,OWL_FLOAT2,mesh.texcoord.size(),
                                mesh.texcoord.data());
      // create the geom
      OWLGeom geom
        = owlGeomCreate(context,triMeshGeomType);

      // set the specific vertex/index buffers required to build the accel
      owlTrianglesSetVertices(geom,vertexBuffer,
                              mesh.vertex.size(),sizeof(vec3f),0);
      owlTrianglesSetIndices(geom,indexBuffer,
                             mesh.index.size(),sizeof(vec3i),0);
      // set sbt data
      owlGeomSetBuffer(geom,"index",indexBuffer);
      owlGeomSetBuffer(geom,"vertex",vertexBuffer);
      owlGeomSetBuffer(geom,"normal",normalBuffer);
      owlGeomSetBuffer(geom,"texcoord",texcoordBuffer);

      owlGeomSet3f(geom,"color",(const owl3f &)mesh.diffuse);
      if (mesh.diffuseTextureID >= 0) {
        owlGeomSet1i(geom,"hasTexture",1);
        assert(mesh.diffuseTextureID < (int)textures.size());
        owlGeomSetTexture(geom,"texture",textures[mesh.diffuseTextureID]);
      } else {
        owlGeomSet1i(geom,"hasTexture",0);
      }
      geoms.push_back(geom);
    }

    OWLGroup triGroup = owlTrianglesGeomGroupCreate(context,geoms.size(),geoms.data());
    owlGroupBuildAccel(triGroup);

    world = owlInstanceGroupCreate(context,1);
    owlInstanceGroupSetChild(world,0,triGroup);
    owlGroupBuildAccel(world);
    owlParamsSetGroup(launchParams,"world",world);
  }

  /*! render one frame */
  void SampleRenderer::render()
  {
    // sanity check: make sure we launch only after first resize is
    // already done:
    if (fbSize.x == 0) return;

    if (!accumulate)
      frameID = 0;
    owlParamsSet1i(launchParams,"frame.frameID",frameID);
    owlParamsSet1i(launchParams,"numPixelSamples",numPixelSamples);
    frameID++;

    owlLaunch2D(rayGen,fbSize.x,fbSize.y,launchParams);
  }

  /*! set camera to render with */
  void SampleRenderer::setCamera(const Camera &camera)
  {
    lastSetCamera = camera;
    // reset accumulation
    frameID = 0;
    owlParamsSet1i(launchParams,"frame.frameID",frameID);
    const vec3f position  = camera.from;
    const vec3f direction = normalize(camera.at-camera.from);

    const float cosFovy = 0.66f;
    const float aspect
      = float(fbSize.x)
      / float(fbSize.y);
    const vec3f horizontal
      = cosFovy * aspect * normalize(cross(direction,
                                           camera.up));
    const vec3f vertical
      = cosFovy * normalize(cross(horizontal,
                                  direction));

    owlParamsSet3f(launchParams,"camera.position",(const owl3f&)position);
    owlParamsSet3f(launchParams,"camera.direction",(const owl3f&)direction);
    owlParamsSet3f(launchParams,"camera.vertical",(const owl3f&)vertical);
    owlParamsSet3f(launchParams,"camera.horizontal",(const owl3f&)horizontal);
  }

  /*! resize frame buffer to given resolution */
  void SampleRenderer::resize(void *fbPointer, const vec2i &newSize)
  {
    if (fbColor) {
      owlBufferDestroy(fbColor);
    }

    this->fbSize = newSize;
    fbColor = owlDeviceBufferCreate(context,OWL_FLOAT4,fbSize.x*fbSize.y,nullptr);

    owlParamsSetBuffer(launchParams,"frame.fbColor",fbColor);
    owlParamsSet1ul(launchParams,"frame.fbFinal",(uint64_t)fbPointer);
    owlParamsSet2i(launchParams,"frame.fbSize",(const owl2i&)fbSize);

    // and re-set the camera, since aspect may have changed
    setCamera(lastSetCamera);
  }

} // ::osc
