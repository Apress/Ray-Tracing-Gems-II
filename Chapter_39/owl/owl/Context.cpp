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

#include "Context.h"
#include "Module.h"
#include "Geometry.h"
#include "Triangles.h"
#include "UserGeom.h"
#include "Texture.h"
#include "TrianglesGeomGroup.h"
#include "UserGeomGroup.h"

#define LOG(message)                            \
  if (Context::logging())                       \
    std::cout                                   \
      << OWL_TERMINAL_LIGHT_BLUE                \
      << "#owl: "                               \
      << message                                \
      << OWL_TERMINAL_DEFAULT << std::endl

#define LOG_OK(message)                         \
  if (Context::logging())                       \
    std::cout                                   \
      << OWL_TERMINAL_BLUE                      \
      << "#owl: "                               \
      << message                                \
      << OWL_TERMINAL_DEFAULT << std::endl

namespace owl {

  Context::Context(int32_t *requestedDeviceIDs,
                   int      numRequestedDevices)
    : buffers(this),
      textures(this),
      groups(this),
      rayGenTypes(this),
      rayGens(this),
      missProgTypes(this),
      missProgs(this),
      geomTypes(this),
      geoms(this),
      modules(this),
      launchParamTypes(this),
      launchParams(this),
      devices(createDeviceContexts(this,
                                   requestedDeviceIDs,
                                   numRequestedDevices))
  {
    enablePeerAccess();

    LaunchParamsType::SP emptyLPType
      = createLaunchParamsType(0,{});
    dummyLaunchParams = createLaunchParams(emptyLPType);
  }
  
  Context::~Context()
  {
    devices.clear();
  }
  

  void Context::enablePeerAccess()
  {
    LOG("enabling peer access ('.'=self, '+'=can access other device)");
    auto &devices = getDevices();

    int deviceCount = int(devices.size());
    LOG("found " << deviceCount << " CUDA capable devices");
    for (auto device : devices) 
      LOG(" - device #" << device->ID << " : " << device->getDeviceName());
    LOG("enabling peer access:");
    
    for (auto device : devices) {
      std::stringstream ss;
      SetActiveGPU forLifeTime(device);
      ss << " - device #" << device->ID << " : ";
      int cuda_i = device->getCudaDeviceID();
      int i = device->ID;
      for (int j=0;j<deviceCount;j++) {
        if (j == i) {
          ss << " ."; 
        } else {
          int cuda_j = devices[j]->getCudaDeviceID();
          int canAccessPeer = 0;
          cudaError_t rc = cudaDeviceCanAccessPeer(&canAccessPeer, cuda_i,cuda_j);
          if (rc != cudaSuccess)
            OWL_RAISE("cuda error in cudaDeviceCanAccessPeer: "
                      +std::to_string(rc));
          if (!canAccessPeer) {
            // huh. this can happen if you have differnt device
            // types (in my case, a 2070 and a rtx 8000).
            // nvm - yup, this isn't an error. Expect certain configs to not allow peer access.
            // disabling this, as it's concerning end users.
            // std::cerr << "cannot not enable peer access!? ... skipping..." << std::endl;
            continue;
          }
          
          rc = cudaDeviceEnablePeerAccess(cuda_j,0);
          if (rc != cudaSuccess)
            OWL_RAISE("cuda error in cudaDeviceEnablePeerAccess: "
                      +std::to_string(rc));
          ss << " +";
        }
      }
      LOG(ss.str()); 
    }
  }
  
  /*! creates a buffer that uses CUDA host pinned memory; that
    memory is pinned on the host and accessive to all devices in the
    device group */
  Buffer::SP
  Context::hostPinnedBufferCreate(OWLDataType type,
                                  size_t count)
  {
    Buffer::SP buffer = std::make_shared<HostPinnedBuffer>(this,type);
    assert(buffer);
    buffer->createDeviceData(getDevices());
    buffer->resize(count);
    return buffer;
  }
  
  /*! creates a buffer that uses CUDA managed memory; that memory is
    managed by CUDA (see CUDAs documentatoin on managed memory) and
    accessive to all devices in the deviec group */
  Buffer::SP
  Context::managedMemoryBufferCreate(OWLDataType type,
                                     size_t count,
                                     const void *init)
  {
    Buffer::SP buffer
      = std::make_shared<ManagedMemoryBuffer>(this,type);
    assert(buffer);
    buffer->createDeviceData(getDevices());
    buffer->resize(count);
    if (init)
      buffer->upload(init, 0, -1);
    return buffer;
  }
  
  Buffer::SP
  Context::deviceBufferCreate(OWLDataType type,
                              size_t count,
                              const void *init)
  {
    Buffer::SP buffer
      = std::make_shared<DeviceBuffer>(this,type);
    assert(buffer);
    buffer->createDeviceData(getDevices());
    buffer->resize(count);
    if (init)
      buffer->upload(init, 0, -1);
    return buffer;
  }

  Texture::SP
  Context::texture2DCreate(OWLTexelFormat texelFormat,
                           OWLTextureFilterMode filterMode,
                           OWLTextureAddressMode addressMode,
                           OWLTextureColorSpace colorSpace,
                           const vec2i size,
                           uint32_t linePitchInBytes,
                           const void *texels)
  {
    Texture::SP texture
      = std::make_shared<Texture>(this,size,linePitchInBytes,
                                  texelFormat,filterMode,addressMode,colorSpace,
                                  texels);
    assert(texture);
    return texture;
  }
    

  Buffer::SP
  Context::graphicsBufferCreate(OWLDataType type,
                                size_t count,
                                cudaGraphicsResource_t resource)
  {
    Buffer::SP buffer
      = std::make_shared<GraphicsBuffer>(this, type, resource);
    
    assert(buffer);
    buffer->createDeviceData(getDevices());
    buffer->resize(count);

    return buffer;
  }

  RayGen::SP
  Context::createRayGen(const std::shared_ptr<RayGenType> &type)
  {
    RayGen::SP rg = std::make_shared<RayGen>(this,type);
    rg->createDeviceData(getDevices());
    return rg;
  }

  LaunchParams::SP
  Context::createLaunchParams(const std::shared_ptr<LaunchParamsType> &type)
  {
    LaunchParams::SP lp = std::make_shared<LaunchParams>(this,type);
    lp->createDeviceData(getDevices());
    return lp;
  }

  MissProg::SP
  Context::createMissProg(const std::shared_ptr<MissProgType> &type)
  {
    MissProg::SP mp = std::make_shared<MissProg>(this,type);
    mp->createDeviceData(getDevices());

    // for backwards compatibility: automatically set miss prog if none are set, yet
    if (mp->ID < numRayTypes &&
        (mp->ID >= (int)missProgPerRayType.size() || !missProgPerRayType[mp->ID]))
      LOG("for backwards compatibility to pre-0.9.0 versions of OWL, "
          "hereby installing this miss program for ray type #" << mp->ID);
      setMissProg(mp->ID,mp);
    return mp;
  }

  /*! sets miss prog to use for a given ray type */
  void Context::setMissProg(int rayTypeToSet, MissProg::SP missProgToUse)
  {
    assert(rayTypeToSet >= 0 && rayTypeToSet < numRayTypes);
    if ((int)missProgPerRayType.size() < numRayTypes)
      missProgPerRayType.resize(numRayTypes);
    missProgPerRayType[rayTypeToSet] = missProgToUse;
  }

  RayGenType::SP
  Context::createRayGenType(Module::SP module,
                            const std::string &progName,
                            size_t varStructSize,
                            const std::vector<OWLVarDecl> &varDecls)
  {
    RayGenType::SP rgt = std::make_shared<RayGenType>(this,
                                                      module,progName,
                                                      varStructSize,
                                                      varDecls);
    rgt->createDeviceData(getDevices());
    return rgt;
  }
  
  
  LaunchParamsType::SP
  Context::createLaunchParamsType(size_t varStructSize,
                                  const std::vector<OWLVarDecl> &varDecls)
  {
    LaunchParamsType::SP lpt
      = std::make_shared<LaunchParamsType>(this,
                                           varStructSize,
                                           varDecls);
    lpt->createDeviceData(getDevices());
    return lpt;
  }
  
  
  MissProgType::SP
  Context::createMissProgType(Module::SP module,
                              const std::string &progName,
                              size_t varStructSize,
                              const std::vector<OWLVarDecl> &varDecls)
  {
    MissProgType::SP mpt
      = std::make_shared<MissProgType>(this,
                                       module,progName,
                                       varStructSize,
                                       varDecls);
    mpt->createDeviceData(getDevices());
    return mpt;
  }
  
  
  GeomGroup::SP Context::trianglesGeomGroupCreate(size_t numChildren, unsigned int buildFlags)
  {
    GeomGroup::SP gg
      = std::make_shared<TrianglesGeomGroup>(this,numChildren,buildFlags);
    gg->createDeviceData(getDevices());
    return gg;
  }
  
  GeomGroup::SP Context::userGeomGroupCreate(size_t numChildren, unsigned int buildFlags)
  {
    GeomGroup::SP gg
      = std::make_shared<UserGeomGroup>(this,numChildren,buildFlags);
    gg->createDeviceData(getDevices());
    return gg;
  }
  
  GeomType::SP
  Context::createGeomType(OWLGeomKind kind,
                          size_t varStructSize,
                          const std::vector<OWLVarDecl> &varDecls)
  {
    GeomType::SP gt;
    switch(kind) {
    case OWL_GEOMETRY_TRIANGLES:
      gt = std::make_shared<TrianglesGeomType>(this,varStructSize,varDecls);
      break;
    case OWL_GEOMETRY_USER:
      gt = std::make_shared<UserGeomType>(this,varStructSize,varDecls);
      break;
    default:
      OWL_NOTIMPLEMENTED;
    }
    gt->createDeviceData(getDevices());
    return gt;
  }

  Module::SP Context::createModule(const std::string &ptxCode)
  {
    Module::SP module = std::make_shared<Module>(this,ptxCode);
    assert(module);
    module->createDeviceData(getDevices());
    return module;
  }

  std::shared_ptr<Geom> UserGeomType::createGeom()
  {
    GeomType::SP self
      = std::dynamic_pointer_cast<GeomType>(shared_from_this());
    Geom::SP geom = std::make_shared<UserGeom>(context,self);
    geom->createDeviceData(context->getDevices());
    return geom;
  }

  std::shared_ptr<Geom> TrianglesGeomType::createGeom()
  {
    GeomType::SP self
      = std::dynamic_pointer_cast<GeomType>(shared_from_this());
    Geom::SP geom = std::make_shared<TrianglesGeom>(context,self);
    geom->createDeviceData(context->getDevices());
    return geom;
  }

  void Context::buildHitGroupRecordsOn(const DeviceContext::SP &device)
  {
    LOG("building SBT hit group records");
    SetActiveGPU forLifeTime(device);
    if (device->sbt.hitGroupRecordsBuffer.alloced())
      device->sbt.hitGroupRecordsBuffer.free();

    size_t maxHitProgDataSize = 0;
    for (size_t i=0;i<geoms.size();i++) {
      Geom *geom = (Geom *)geoms.getPtr(i);
      if (!geom) continue;
      
      assert(geom->geomType);
      maxHitProgDataSize = std::max(maxHitProgDataSize,geom->geomType->varStructSize);
    }
      
    size_t numHitGroupEntries = sbtRangeAllocator.maxAllocedID;
    // always add 1 so we always have a hit group array, even for
    // programs that didn't create any Groups (yet?)
    size_t numHitGroupRecords = numHitGroupEntries*numRayTypes + 1;
    size_t hitGroupRecordSize
      = OPTIX_SBT_RECORD_HEADER_SIZE
      + smallestMultipleOf<OPTIX_SBT_RECORD_ALIGNMENT>(maxHitProgDataSize);
    
    assert((OPTIX_SBT_RECORD_HEADER_SIZE % OPTIX_SBT_RECORD_ALIGNMENT) == 0);
    device->sbt.hitGroupRecordSize = hitGroupRecordSize;
    device->sbt.hitGroupRecordCount = numHitGroupRecords;
    
    size_t totalHitGroupRecordsArraySize
      = numHitGroupRecords * hitGroupRecordSize;
    std::vector<uint8_t> hitGroupRecords(totalHitGroupRecordsArraySize);

    // ------------------------------------------------------------------
    // now, write all records (only on the host so far): we need to
    // write one record per geometry, per ray type
    // ------------------------------------------------------------------
    for (size_t groupID=0;groupID<groups.size();groupID++) {
      Group *group = groups.getPtr(groupID);
      if (!group) continue;
      GeomGroup *gg = dynamic_cast<GeomGroup *>(group);
      if (!gg) continue;
        
      const size_t sbtOffset = gg->sbtOffset;
      for (size_t childID=0;childID<gg->geometries.size();childID++) {
        Geom::SP geom = gg->geometries[childID];
        if (!geom) continue;
          
        // const int geomID    = geom->ID;
        for (int rayTypeID=0;rayTypeID<numRayTypes;rayTypeID++) {
          // ------------------------------------------------------------------
          // compute pointer to entire record:
          // ------------------------------------------------------------------
          const size_t recordID
            = (sbtOffset+childID)*numRayTypes + rayTypeID;
          assert(recordID < numHitGroupRecords);
          uint8_t *const sbtRecord
            = hitGroupRecords.data() + recordID*hitGroupRecordSize;

          // let the geometry write itself:
          geom->writeSBTRecord(sbtRecord,device,rayTypeID);
          
        }
      }
    }
    device->sbt.hitGroupRecordsBuffer.alloc(hitGroupRecords.size());
    device->sbt.hitGroupRecordsBuffer.upload(hitGroupRecords);
    
    LOG_OK("done building (and uploading) SBT hit group records");
  }
  
  
  void Context::buildMissProgRecordsOn(const DeviceContext::SP &device)
  {
    LOG("building SBT miss group records");
    SetActiveGPU forLifeTime(device);
    
    size_t numMissProgRecords = numRayTypes;
    if ((int)missProgPerRayType.size() < numRayTypes)
      missProgPerRayType.resize(numRayTypes);
    
    size_t maxMissProgDataSize = 0;
    for (int i=0;i<(int)missProgPerRayType.size();i++) {
      MissProg::SP missProg = missProgPerRayType[i];
      if (!missProg) continue;
      assert(missProg->type);
      maxMissProgDataSize = std::max(maxMissProgDataSize,missProg->type->varStructSize);
    }
    
    size_t missProgRecordSize
      = OPTIX_SBT_RECORD_HEADER_SIZE
      + smallestMultipleOf<OPTIX_SBT_RECORD_ALIGNMENT>(maxMissProgDataSize);
    
    assert((OPTIX_SBT_RECORD_HEADER_SIZE % OPTIX_SBT_RECORD_ALIGNMENT) == 0);
    device->sbt.missProgRecordSize  = missProgRecordSize;
    device->sbt.missProgRecordCount = numMissProgRecords;

    size_t totalMissProgRecordsArraySize
      = numMissProgRecords * missProgRecordSize;
    std::vector<uint8_t> missProgRecords(totalMissProgRecordsArraySize);

    // ------------------------------------------------------------------
    // now, write all records (only on the host so far): we need to
    // write one record per geometry, per ray type
    // ------------------------------------------------------------------
    for (size_t recordID=0;recordID<numMissProgRecords;recordID++) {
      MissProg::SP miss = missProgPerRayType[recordID];
      if (!miss) continue;
      
      uint8_t *const sbtRecord
        = missProgRecords.data() + recordID*missProgRecordSize;
      miss->writeSBTRecord(sbtRecord,device);
    }
    device->sbt.missProgRecordsBuffer.alloc(missProgRecords.size());
    device->sbt.missProgRecordsBuffer.upload(missProgRecords);
    LOG_OK("done building (and uploading) SBT miss group records");
  }


  void Context::buildRayGenRecordsOn(const DeviceContext::SP &device)
  {
    LOG("building SBT rayGen prog records");
    SetActiveGPU forLifeTime(device);

    for (size_t rgID=0;rgID<rayGens.size();rgID++) {
      auto rg = rayGens.getPtr(rgID);
      assert(rg);
      auto &dd = rg->getDD(device);
      
      std::vector<uint8_t> hostMem(dd.rayGenRecordSize);
      rg->writeSBTRecord(hostMem.data(),device);
      dd.sbtRecordBuffer.upload(hostMem);
    }
  }
  
  void Context::buildSBT(OWLBuildSBTFlags flags)
  {
    if (flags & OWL_SBT_HITGROUPS)
      for (auto device : getDevices())
        buildHitGroupRecordsOn(device);
    
    // ----------- build miss prog(s) -----------
    if (flags & OWL_SBT_MISSPROGS)
      for (auto device : getDevices())
        buildMissProgRecordsOn(device);

    // ----------- build raygens -----------
    if (flags & OWL_SBT_RAYGENS)
      for (auto device : getDevices())
        buildRayGenRecordsOn(device);
  }

  void Context::buildPipeline()
  {
    for (auto device : getDevices()) {
      device->destroyPipeline();
      device->buildPipeline();
    }
  }
  
  void Context::buildModules(bool debug)
  {
    destroyModules();
    for (auto device : getDevices()) {
      device->configurePipelineOptions(debug);
      for (int moduleID=0;moduleID<(int)modules.size();moduleID++) {
        Module *module = modules.getPtr(moduleID);
        if (!module) continue;
        
        module->getDD(device).build();
      }
    }
  }
  
  void Context::setRayTypeCount(size_t rayTypeCount)
  {
    /* TODO; sanity checking that this is a useful value, and that
       no geoms etc are created yet */
    this->numRayTypes = int(rayTypeCount);
  }

  void Context::setBoundLaunchParamValues(const std::vector<OWLBoundValueDecl> &boundValues)
  {
#if OPTIX_VERSION >= 70200
    this->boundLaunchParamValues.clear();
    this->boundLaunchParamValues.reserve(boundValues.size());
    for (const OWLBoundValueDecl &v : boundValues) {
      this->boundLaunchParamValues.push_back( {
        v.var.offset,
        sizeOf(v.var.type),
        v.boundValuePtr });
    }
#else
    LOG("Ignoring bound launch params for old version of OptiX");
#endif
  }

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
  void Context::setMaxInstancingDepth(int32_t maxInstanceDepth)
  {
    this->maxInstancingDepth = maxInstanceDepth;
    
    if (maxInstancingDepth < 1)
      OWL_RAISE
        ("a instancing depth of < 1 isnt' currently supported in OWL; "
         "please see comments on owlSetMaxInstancingDepth() (owl/owl_host.h)");
    
    for (auto device : getDevices()) {
      assert("check pipeline isn't already created"
             && device->pipeline == nullptr);
      
      device->configurePipelineOptions();
    } 
  }

  void Context::enableMotionBlur()
  {
    motionBlurEnabled = true;
  }

  void Context::setNumAttributeValues(size_t numAttributeValues)
  {
    for (auto device : getDevices()) {
      assert("check programs have not been built"
             && device->allActivePrograms.empty());
    }
    this->numAttributeValues = (int)numAttributeValues;
  }

  void Context::buildPrograms(bool debug)
  {
    buildModules(debug);
    
    for (auto device : getDevices()) {
      SetActiveGPU forLifeTime(device);
      device->buildPrograms();
    }
  }


  void Context::destroyModules()
  {
    for (size_t moduleID=0;moduleID<modules.size();moduleID++) {
      Module *module = modules.getPtr(moduleID);
      if (module)
        for (auto device : getDevices())
          module->getDD(device).destroy();
    }
  }
    
  void Context::destroyPrograms()
  {
    for (auto device : getDevices()) 
      device->destroyPrograms();
  }

} // ::owl
