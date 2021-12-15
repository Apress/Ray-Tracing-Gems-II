// ======================================================================== //
// Copyright 2019-2021 Ingo Wald                                            //
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

#include "DeviceContext.h"
#include "ObjectRegistry.h"
#include "Buffer.h"
#include "Texture.h"
#include "Group.h"
#include "RayGen.h"
#include "LaunchParams.h"
#include "MissProg.h"

namespace owl {

  /*! the root 'context' that spans, and manages, all objects and all
    devices */
  struct Context : public Object {
    typedef std::shared_ptr<Context> SP;

    /*! returns whether logging is enabled */
    inline static bool logging()
    {
#ifdef NDEBUG
      return false;
#else
      return true;
#endif
    }

    /*! pretty-printer, for printf-debugging */
    virtual std::string toString() const override { return "owl::Context"; }

    /*! creates a context with the given device IDs. If list of device
      is nullptr, and number requested devices is > 1, then the
      first N devices will get used; invalid device IDs in the list
      will automatically get dropped */
    Context(int32_t *requestedDeviceIDs,
            int      numRequestedDevices);

    /*! virtual destructor to cleanly wind down upon exit */
    virtual ~Context();

    size_t deviceCount() const { return getDevices().size(); }
    const std::vector<DeviceContext::SP> &getDevices() const { return devices; }
    DeviceContext::SP getDevice(int ID) const
    { assert(ID >= 0 && ID < (int)devices.size()); return devices[ID]; }

    /*! part of the SBT creation - builds the hit group array */
    void buildHitGroupRecordsOn(const DeviceContext::SP &device);
    /*! part of the SBT creation - builds the raygen array */
    void buildRayGenRecordsOn(const DeviceContext::SP &device);
    /*! part of the SBT creation - builds the miss group array */
    void buildMissProgRecordsOn(const DeviceContext::SP &device);

    /*! sets number of ray types to be used - should be done right
      after context creation, and before SBT and pipeline get
      built */
    void setRayTypeCount(size_t rayTypeCount);

    void setBoundLaunchParamValues(const std::vector<OWLBoundValueDecl> &boundValues);
    
    /*! enables motoin blur - should be done right after context
      creation, and before SBT and pipeline get built */
    void enableMotionBlur();
    
    /*! sets maximum instancing depth for the given context:

      '0' means 'no instancing allowed, only bottom-level accels; 
  
      '1' means 'at most one layer of instances' (ie, a two-level scene),
      where the 'root' world rays are traced against can be an instance
      group, but every child in that inscne group is a geometry group.

      'N>1" means "up to N layers of instances are allowed.

      The default instancing depth is 1 (ie, a two-level scene), since
      this allows for most use cases of instancing and is still
      hardware-accelerated. Using a node graph with instancing deeper than
      the configured value will result in wrong results; but be aware that
      using any value > 1 here will come with a cost. It is recommended
      to, if at all possible, leave this value to one and convert the
      input scene to a two-level scene layout (ie, with only one level of
      instances) */
    void setMaxInstancingDepth(int32_t maxInstanceDepth);

    /* Set number of attributes for passing data from custom Intersection programs
       to ClosestHit programs.  Default 2.  Has no effect once programs are built.*/
    void setNumAttributeValues(size_t numAttributeValues);


    // ------------------------------------------------------------------
    // internal mechanichs/plumbling that do the actual work
    // ------------------------------------------------------------------
    
    void buildSBT(OWLBuildSBTFlags flags);
    void buildPipeline();
    void buildPrograms(bool debug = false);
    /*! clearly destroy _pptix_ handles of all active programs */
    void destroyPrograms();
    void buildModules(bool debug = false);
    /*! clearly destroy _optix_ handles of all active modules */
    void destroyModules();


    // ------------------------------------------------------------------
    // factory methods to create objects within this context
    // ------------------------------------------------------------------

    /*! creates a 2D texture object with given parameters; this will
      internally be mapped to a cuda texture object, and uploaded as
      such to the device */
    Texture::SP
    texture2DCreate(OWLTexelFormat texelFormat,
                    OWLTextureFilterMode filterMode,
                    OWLTextureAddressMode addressMode,
                    OWLTextureColorSpace colorSpace,
                    const vec2i size,
                    uint32_t linePitchInBytes,
                    const void *texels);

    /*! create a new *triangles* geometry group that will eventually
      create a BVH over all the trinalges in all its child
      geometries. only TrianglesGeoms can be added to this
      group. These triangle geoms can all have different types,
      different programs, etc, but must all be of "OWL_TRIANGLES"
      kind */
    GeomGroup::SP
    trianglesGeomGroupCreate(size_t numChildren, unsigned int buildFlags);
    
    /*! create a new *user* geometry group that will eventually create
      a BVH over all the user geoms / custom prims in all its child
      geometries. only UserGeom's can be added to this group. These
      user geoms can all have different types, different programs,
      etc, but must all be of "OWL_GEOMETRY_USER" kind */
    GeomGroup::SP
    userGeomGroupCreate(size_t numChildren, unsigned int buildFlags);

    /*! create a new device buffer of given data type and count; if
      init is non-null it will be used to populoate this
      buffer. Note that for certain non-trivial types (OWLTexture,
      OWLGroup, etc) you may have to specify the content upon
      creation */
    Buffer::SP
    deviceBufferCreate(OWLDataType type,
                       size_t count,
                       const void *init);

    /*! creates a buffer that uses CUDA host pinned memory; that
      memory is pinned on the host and accessive to all devices in the
      device group */
    Buffer::SP
    hostPinnedBufferCreate(OWLDataType type,
                           size_t count);
    
    /*! creates a buffer that uses CUDA managed memory; that memory is
      managed by CUDA (see CUDAs documentation on managed memory) and
      accessive to all devices in the deviec group */
    Buffer::SP
    managedMemoryBufferCreate(OWLDataType type,
                              size_t count,
                              const void *init);

    /*! creates a buffer that wraps a CUDA graphics resource
      that can be, for instance, an OpenGL texture */
    Buffer::SP
    graphicsBufferCreate(OWLDataType type,
                         size_t count,
                         cudaGraphicsResource_t resource);

    
    /*! creates new ray gen program *type* with given program name (in
      given module), and the given variable declarations that
      describe this type's variables */
    RayGenType::SP
    createRayGenType(Module::SP module,
                     const std::string &progName,
                     size_t varStructSize,
                     const std::vector<OWLVarDecl> &varDecls);

    /*! create new instance of a ray gen program of given type */
    RayGen::SP
    createRayGen(const std::shared_ptr<RayGenType> &type);
    
    /*! create a new launch param type descriptor with given
      variables; this can then be used to create actual launch param
      instances (\see createLaunchParams) */
    LaunchParamsType::SP
    createLaunchParamsType(size_t varStructSize,
                           const std::vector<OWLVarDecl> &varDecls);
    
    /*! create new instance of a set of launch params of given type */
    LaunchParams::SP
    createLaunchParams(const std::shared_ptr<LaunchParamsType> &type);
    
    /*! creates new miss program *type* with given program name (in
      given module), and the given variable declarations that
      describe this type's variables */
    MissProgType::SP
    createMissProgType(Module::SP module,
                       const std::string &progName,
                       size_t varStructSize,
                       const std::vector<OWLVarDecl> &varDecls);

    /*! create new instance of a miss program of given type */
    MissProg::SP
    createMissProg(const std::shared_ptr<MissProgType> &type);

    /*! sets miss prog to use for a given ray type */
    void setMissProg(int rayTypeToSet, MissProg::SP missProgToUse);
    
    /*! creates new geometry type defitiion with given variable declarations */
    GeomType::SP
    createGeomType(OWLGeomKind kind,
                   size_t varStructSize,
                   const std::vector<OWLVarDecl> &varDecls);
    
    /*! creates new module with given precompiled PTX code */
    Module::SP
    createModule(const std::string &ptxCode);

    // ------------------------------------------------------------------
    // member variables
    // ------------------------------------------------------------------

    /*! @{ registries for all the different object types within this
      context. allows for keeping track what's alive, and what has
      to be compiled, put into SBTs, etc */
    ObjectRegistryT<Buffer>       buffers;
    ObjectRegistryT<Texture>      textures;
    ObjectRegistryT<Group>        groups;
    ObjectRegistryT<RayGenType>   rayGenTypes;
    ObjectRegistryT<RayGen>       rayGens;
    ObjectRegistryT<MissProgType> missProgTypes;
    ObjectRegistryT<MissProg>     missProgs;
    ObjectRegistryT<GeomType>     geomTypes;
    ObjectRegistryT<Geom>         geoms;
    ObjectRegistryT<Module>       modules;
    ObjectRegistryT<LaunchParamsType> launchParamTypes;
    ObjectRegistryT<LaunchParams>     launchParams;
    /*! @} */

    /*! tracks which ID regions in the SBT have already been used -
      newly created groups allocate ranges of IDs in the SBT (to
      allow its geometries to be in successive SBT regions), and
      this struct keeps track of whats already used, and what is
      available */
    RangeAllocator sbtRangeAllocator;

    /*! one miss prog per ray type */
    std::vector<MissProg::SP> missProgPerRayType;

    /*! maximum depth instancing tree as specified by
      `setMaxInstancingDepth` */
    int maxInstancingDepth = 1;

    /*! number of ray types - change via setRayTypeCount() */
    int numRayTypes { 1 };

#if OPTIX_VERSION >= 70200
    /*! bound values of launch params, for specializing modules during compile */
    std::vector<OptixModuleCompileBoundValueEntry> boundLaunchParamValues;
#endif
    
    /*! by default motion blur is off, as it costs performacne - set
      via enableMotimBlur() */
    bool motionBlurEnabled = false;

    /* Number of attributes for writing data between Intersection and ClosestHit */
    int numAttributeValues = 2;

    /*! a set of dummy (ie, empty) launch params. allows us for always
      using the same launch code, *with* launch params, even if th
      user didn't specify any during launch */
    LaunchParams::SP dummyLaunchParams;

  private:
    void enablePeerAccess();
    std::vector<DeviceContext::SP> devices;
  };

} // ::owl

