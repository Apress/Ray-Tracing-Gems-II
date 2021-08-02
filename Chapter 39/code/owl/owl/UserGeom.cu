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

#include "UserGeom.h"
#include "Context.h"

namespace owl {

#define LOG(message)                                    \
  if (Context::logging())                               \
    std::cout << "#owl(" << device->ID << "): "         \
              << message                                \
              << std::endl

#define LOG_OK(message)                                         \
  if (Context::logging())                                       \
    std::cout << OWL_TERMINAL_GREEN                             \
              << "#owl(" << device->ID << "): "                 \
              << message << OWL_TERMINAL_DEFAULT << std::endl

  __device__ static float atomicMax(float* address, float val)
  {
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
      assumed = old;
      old = ::atomicCAS(address_as_i, assumed,
                        __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
  }
  
  __device__ static float atomicMin(float* address, float val)
  {
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
      assumed = old;
      old = ::atomicCAS(address_as_i, assumed,
                        __float_as_int(::fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
  }
  
  __global__ void computeBoundsOfPrimBounds(box3f *d_bounds,
                                            const box3f *d_primBounds,
                                            size_t count)
  {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= count) return;

    box3f box = d_primBounds[tid];
    if (!box.empty()) {
      atomicMin(&d_bounds->lower.x,box.lower.x);
      atomicMin(&d_bounds->lower.y,box.lower.y);
      atomicMin(&d_bounds->lower.z,box.lower.z);
      atomicMax(&d_bounds->upper.x,box.upper.x);
      atomicMax(&d_bounds->upper.y,box.upper.y);
      atomicMax(&d_bounds->upper.z,box.upper.z);
    }
  }
                                          
  /*! construct a new device-data for this type */
  UserGeomType::DeviceData::DeviceData(const DeviceContext::SP &device)
    : GeomType::DeviceData(device)
  {}

  /*! creates the device-specific data for this group */
  RegisteredObject::DeviceData::SP
  UserGeom::createOn(const DeviceContext::SP &device) 
  {
    return std::make_shared<DeviceData>(device);
  }

  /*! create this object's device-specific data for the device */
  RegisteredObject::DeviceData::SP
  UserGeomType::createOn(const DeviceContext::SP &device) 
  {
    return std::make_shared<DeviceData>(device);
  }

  /*! pretty-printer, for printf-debugging */
  std::string UserGeomType::toString() const
  { return "UserGeomType"; }

  /*! pretty-printer, for printf-debugging */
  std::string UserGeom::toString() const
  { return "UserGeom"; }
  
  /*! call a cuda kernel that computes the bounds of the vertex
      buffers. note we alwyas (and only) do this on the first GPU */
  void UserGeom::computeBounds(box3f bounds[2])
  {
    int numThreads = 1024;
    int numBlocks = int((primCount + numThreads - 1) / numThreads);

    DeviceContext::SP device = context->getDevices()[0];
    SetActiveGPU forLifeTime(device);

    DeviceMemory d_bounds;
    d_bounds.alloc(sizeof(box3f));
    bounds[0] = bounds[1] = box3f();
    d_bounds.upload(bounds);

    DeviceData &dd = getDD(device);
    
    computeBoundsOfPrimBounds<<<numBlocks,numThreads>>>
      (((box3f*)d_bounds.get())+0,
       (box3f *)dd.internalBufferForBoundsProgram.get(),
       primCount);
    CUDA_SYNC_CHECK();
    d_bounds.download(&bounds[0]);
    d_bounds.free();
    CUDA_SYNC_CHECK();
    bounds[1] = bounds[0];
  }

  UserGeomType::UserGeomType(Context *const context,
                             size_t varStructSize,
                             const std::vector<OWLVarDecl> &varDecls)
    : GeomType(context,varStructSize,varDecls),
      intersectProg(context->numRayTypes)
  {
    /*! nothing special - all inherited */
  }

  /*! constructor */
  UserGeom::UserGeom(Context *const context,
                     GeomType::SP geometryType)
    : Geom(context,geometryType)
  {}

  /*! set number of primitives that this geom will contain */
  void UserGeom::setPrimCount(size_t count)
  {
    primCount = count;
  }

  /*! set intersection program to run for this type and given ray type */
  void UserGeomType::setIntersectProg(int rayType,
                                      Module::SP module,
                                      const std::string &progName)
  {
    assert(rayType >= 0 && rayType < intersectProg.size());

    intersectProg[rayType].progName = "__intersection__"+progName;
    intersectProg[rayType].module   = module;
  }

  /*! set bounding box program to run for this type */
  void UserGeomType::setBoundsProg(Module::SP module,
                                   const std::string &progName)
  {
    this->boundsProg.progName = progName;
    this->boundsProg.module   = module;
  }

  /*! run the bounding box program for all primitives within this geometry */
  void UserGeom::executeBoundsProgOnPrimitives(const DeviceContext::SP &device)
  {
    SetActiveGPU activeGPU(device);
      
    std::vector<uint8_t> userGeomData(geomType->varStructSize);
    
    DeviceMemory tempMem;
    tempMem.alloc(geomType->varStructSize);
    
    DeviceData &dd = getDD(device);
    dd.internalBufferForBoundsProgram.allocManaged(primCount*sizeof(box3f));

    writeVariables(userGeomData.data(),device);
        
    // size of each thread block during bounds function call
    vec3i blockDims(32,32,1);
    uint32_t threadsPerBlock = blockDims.x*blockDims.y*blockDims.z;
        
    uint32_t numBlocks = owl::common::divRoundUp((uint32_t)primCount,threadsPerBlock);
    uint32_t numBlocks_x
      = 1+uint32_t(powf((float)numBlocks,1.f/3.f));
    uint32_t numBlocks_y
      = 1+uint32_t(sqrtf((float)(numBlocks/numBlocks_x)));
    uint32_t numBlocks_z
      = owl::common::divRoundUp(numBlocks,numBlocks_x*numBlocks_y);
        
    vec3i gridDims(numBlocks_x,numBlocks_y,numBlocks_z);

    tempMem.upload(userGeomData);
    
    void  *d_geomData = tempMem.get();
    vec3f *d_boundsArray = (vec3f*)dd.internalBufferForBoundsProgram.get();
    /* arguments for the kernel we are to call */
    void  *args[] = {
                     &d_geomData,
                     &d_boundsArray,
                     (void *)&primCount
    };
    
    CUstream stream = device->stream;
    UserGeomType::DeviceData &typeDD = getTypeDD(device);
    if (!typeDD.boundsFuncKernel)
      throw std::runtime_error("bounds kernel set, but not yet compiled - "
                               "did you forget to call BuildPrograms() before"
                               " (User)GroupAccelBuild()!?");
        
    CUresult rc
      = cuLaunchKernel(typeDD.boundsFuncKernel,
                       gridDims.x,gridDims.y,gridDims.z,
                       blockDims.x,blockDims.y,blockDims.z,
                       0, stream, args, 0);
    if (rc) {
      const char *errName = 0;
      cuGetErrorName(rc,&errName);
      throw std::runtime_error("unknown CUDA error in calling bounds function kernel: "
                               +std::string(errName));
    }
    
    tempMem.free();
    cudaDeviceSynchronize();
  }

  /*! fill in an OptixProgramGroup descriptor with the module and
    program names for this type */
  void UserGeomType::DeviceData::fillPGDesc(OptixProgramGroupDesc &pgDesc,
                                            GeomType *_parent,
                                            int rt)
  {
    GeomType::DeviceData::fillPGDesc(pgDesc,_parent,rt);
    UserGeomType *parent = (UserGeomType*)_parent;
    
    // ----------- intserect -----------
    if (rt < (int)parent->intersectProg.size()) {
      const ProgramDesc &pd = parent->intersectProg[rt];
      if (pd.module) {
        pgDesc.hitgroup.moduleIS = pd.module->getDD(device).module;
        pgDesc.hitgroup.entryFunctionNameIS = pd.progName.c_str();
      }
    }
  }
  
  /*! build the CUDA bounds program kernel (if bounds prog is set) */
  void UserGeomType::buildBoundsProg()
  {
    if (!boundsProg.module) return;
    
    Module::SP module = boundsProg.module;
    assert(module);

    for (auto device : context->getDevices()) {
      LOG("building bounds function ....");
      SetActiveGPU forLifeTime(device);
      auto &typeDD = getDD(device);
      auto &moduleDD = module->getDD(device);
      
      assert(moduleDD.boundsModule);

      const std::string annotatedProgName
        = std::string("__boundsFuncKernel__")
        + boundsProg.progName;
    
      CUresult rc = cuModuleGetFunction(&typeDD.boundsFuncKernel,
                                        moduleDD.boundsModule,
                                        annotatedProgName.c_str());
      
      switch(rc) {
      case CUDA_SUCCESS:
        /* all OK, nothing to do */
        LOG_OK("found bounds function " << annotatedProgName << " ... perfect!");
        break;
      case CUDA_ERROR_NOT_FOUND:
        throw std::runtime_error("in "+std::string(__PRETTY_FUNCTION__)
                                 +": could not find OPTIX_BOUNDS_PROGRAM("
                                 +boundsProg.progName+")");
      default:
        const char *errName = 0;
        cuGetErrorName(rc,&errName);
        throw std::runtime_error("unknown CUDA error when building bounds program kernel"
                                 +std::string(errName));
      }
    }
  }

} // ::owl
