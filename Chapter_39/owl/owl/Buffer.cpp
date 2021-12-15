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

#include "Buffer.h"
#include "Context.h"
#include "APIHandle.h"
#include "owl/owl_device_buffer.h"

namespace owl {

  // ------------------------------------------------------------------
  // Buffer::DeviceData
  // ------------------------------------------------------------------
  
  /*! constructor */
  Buffer::DeviceData::DeviceData(const DeviceContext::SP &device)
    : RegisteredObject::DeviceData(device)
  {}

  // ------------------------------------------------------------------
  // Buffer
  // ------------------------------------------------------------------
  
  Buffer::Buffer(Context *const context,
                 OWLDataType type)
    : RegisteredObject(context,context->buffers),
      type(type)
  {
  }

  Buffer::~Buffer()
  {
    destroy();
  }
  
  /*! pretty-printer, for printf-debugging */
  std::string Buffer::toString() const 
  {
    return "Buffer";
  }

  /*! destroy whatever resouces this buffer's ll-layer handle this
    may refer to; this will not destruct the current object
    itself, but should already release all its references */
  void Buffer::destroy()
  {
    if (ID < 0)
      /* already destroyed */
      return;

    deviceData.clear();
    
    registry.forget(this); // sets ID to -1
  }
  
  /*! creates the device-specific data for this group */
  RegisteredObject::DeviceData::SP Buffer::createOn(const DeviceContext::SP &device)
  {
    return std::make_shared<Buffer::DeviceData>(device);
  }

  // ------------------------------------------------------------------
  // Device Buffer
  // ------------------------------------------------------------------
  
  /*! any device-specific data, such as optix handles, cuda device
    pointers, etc */
  DeviceBuffer::DeviceData::DeviceData(DeviceBuffer *parent, const DeviceContext::SP &device)
    : Buffer::DeviceData(device), parent(parent)
  {}

  /*! pretty-printer, for debugging */
  std::string DeviceBuffer::toString() const 
  {
    return "DeviceBuffer";
  }

  DeviceBuffer::DeviceData::~DeviceData()
  {
    if (d_pointer == 0) return;

    SetActiveGPU forLifeTime(device);

    OWL_CUDA_CALL_NOTHROW(Free(d_pointer));
    d_pointer = nullptr;
  }
  
  /*! creates the device-specific data for this group */
  RegisteredObject::DeviceData::SP DeviceBuffer::createOn(const DeviceContext::SP &device)
  {
    if (type >= _OWL_BEGIN_COPYABLE_TYPES)
      return std::make_shared<DeviceBuffer::DeviceDataForCopyableData>(this,device);

    if (type == OWL_BUFFER)
      return std::make_shared<DeviceBuffer::DeviceDataForBuffers>(this,device);

    if (type == OWL_GROUP)
      return std::make_shared<DeviceBuffer::DeviceDataForGroups>(this,device);

    if (type == OWL_TEXTURE)
      return std::make_shared<DeviceBuffer::DeviceDataForTextures>(this,device);

    OWL_RAISE("unsupported element type for device buffer");
    return nullptr;
  }
  
  void DeviceBuffer::clear()
  {
    for (auto dd : deviceData)
      dd->as<DeviceBuffer::DeviceData>().clear();
    OWL_CUDA_SYNC_CHECK();
  }
  
  void DeviceBuffer::upload(const void *hostPtr, size_t offset, int64_t count)
  {
    assert(deviceData.size() == context->deviceCount());
    for (auto dd : deviceData)
      dd->as<DeviceBuffer::DeviceData>().uploadAsync(hostPtr, offset, count);
    OWL_CUDA_SYNC_CHECK();
  }
  
  void DeviceBuffer::upload(const int deviceID, const void *hostPtr, size_t offset, int64_t count) 
  {
    assert(deviceID < (int)deviceData.size());
    deviceData[deviceID]->as<DeviceBuffer::DeviceData>().uploadAsync(hostPtr, offset, count);
    OWL_CUDA_SYNC_CHECK();
  }
  

  DeviceBuffer::DeviceBuffer(Context *const context,
                             OWLDataType type)
    : Buffer(context,type)
  {}

  void DeviceBuffer::resize(size_t newElementCount)
  {
    elementCount = newElementCount;
    for (auto device : context->getDevices()) 
      getDD(device).executeResize();
  }
  
  void DeviceBuffer::DeviceDataForTextures::executeResize() 
  {
    SetActiveGPU forLifeTime(device);
    if (d_pointer) {
      OWL_CUDA_CALL(Free(d_pointer)); d_pointer = nullptr;
    }

    if (parent->elementCount)
      OWL_CUDA_CALL(Malloc(&d_pointer,parent->elementCount*sizeof(cudaTextureObject_t)));
  }

  void DeviceBuffer::DeviceDataForTextures::clear() 
  {
    throw std::runtime_error("owlBufferClear() not implmemented for buffers of textures");
  }
  
  void DeviceBuffer::DeviceDataForTextures::uploadAsync(const void *hostDataPtr, size_t offset, int64_t count) 
  {
    SetActiveGPU forLifeTime(device);
    
    hostHandles.resize((count == -1) ? parent->elementCount : count);
    APIHandle **apiHandles = (APIHandle **)hostDataPtr;
    std::vector<cudaTextureObject_t> devRep((count == -1) ? parent->elementCount : count);
    
    for (size_t i=0; i < ((count == -1) ? parent->elementCount : count); i++)
      if (apiHandles[i]) {
        Texture::SP texture = apiHandles[i]->object->as<Texture>();
        assert(texture && "make sure those are really textures in this buffer!");
        devRep[i] = texture->textureObjects[device->ID];
        hostHandles[i] = texture;
      } else
        hostHandles[i] = nullptr;

    OWL_CUDA_CALL(MemcpyAsync((char*)d_pointer + offset, devRep.data(),
                          devRep.size()*sizeof(devRep[0]),
                          cudaMemcpyDefault,
                          device->getStream()));
  }
  
  void DeviceBuffer::DeviceDataForBuffers::clear() 
  {
    throw std::runtime_error("owlBufferClear() not implmemented for buffers of buffers");
  }
  
  void DeviceBuffer::DeviceDataForBuffers::executeResize() 
  {
    SetActiveGPU forLifeTime(device);
    
    if (d_pointer) {
      OWL_CUDA_CALL(Free(d_pointer)); d_pointer = nullptr;
    }

    if (parent->elementCount) {
      OWL_CUDA_CALL(Malloc(&d_pointer,parent->elementCount*sizeof(device::Buffer)));
    }
  }
  
  void DeviceBuffer::DeviceDataForBuffers::uploadAsync(const void *hostDataPtr, size_t offset, int64_t count) 
  {
    SetActiveGPU forLifeTime(device);
    
    hostHandles.resize( (count == -1) ? parent->elementCount : count);
    APIHandle **apiHandles = (APIHandle **)hostDataPtr;
    std::vector<device::Buffer> devRep( (count == -1) ? parent->elementCount : count);
    
    for (int i=0; i < int((count == -1) ? parent->elementCount : count); i++)
      if (apiHandles[i]) {
        Buffer::SP buffer = apiHandles[i]->object->as<Buffer>();
        assert(buffer && "make sure those are really textures in this buffer!");
        
        devRep[i].data    = (void*)buffer->getPointer(device);
        devRep[i].type    = buffer->type;
        devRep[i].count   = buffer->getElementCount();
        
        hostHandles[i] = buffer;
      } else {
        devRep[i].data    = 0;
        devRep[i].type    = OWL_INVALID_TYPE;
        devRep[i].count   = 0;
      }

    OWL_CUDA_CALL(MemcpyAsync((char*)d_pointer + offset,devRep.data(),
                          devRep.size()*sizeof(devRep[0]),
                          cudaMemcpyDefault,
                          device->getStream()));
  }




  void DeviceBuffer::DeviceDataForGroups::clear() 
  {
    throw std::runtime_error("owlBufferClear() not implmemented for buffers of groups");
  }
  

  void DeviceBuffer::DeviceDataForGroups::executeResize() 
  {
    SetActiveGPU forLifeTime(device);
    
    if (d_pointer) { OWL_CUDA_CALL(Free(d_pointer)); d_pointer = nullptr; }

    if (parent->elementCount)
      OWL_CUDA_CALL(Malloc(&d_pointer,parent->elementCount*sizeof(OptixTraversableHandle)));
  }
  
  void DeviceBuffer::DeviceDataForGroups::uploadAsync(const void *hostDataPtr, size_t offset, int64_t count) 
  {
    SetActiveGPU forLifeTime(device);
    
    hostHandles.resize( (count == -1) ? parent->elementCount : count);
    APIHandle **apiHandles = (APIHandle **)hostDataPtr;
    std::vector<OptixTraversableHandle> devRep( (count == -1) ? parent->elementCount : count);
    
    for (int i=0; i < int((count == -1) ? parent->elementCount : count); i++)
      if (apiHandles[i]) {
        Group::SP group = apiHandles[i]->object->as<Group>();
        assert(group && "make sure those are really groups in this buffer!");

        devRep[i] 
          = group->getTraversable(device);
        // devRep[i].data    = (void*)buffer->getPointer(device);
        // devRep[i].type    = buffer->type;
        // devRep[i].count   = buffer->getElementCount();
        
        hostHandles[i] = group;
      } else {
        devRep[i] = 0;
      }

    OWL_CUDA_CALL(MemcpyAsync((char*)d_pointer + offset,devRep.data(),
                          devRep.size()*sizeof(devRep[0]),
                          cudaMemcpyDefault,
                          device->getStream()));
  }


  void DeviceBuffer::DeviceDataForCopyableData::clear() 
  {
    SetActiveGPU forLifeTime(device);
    
    if (parent->elementCount) {
      OWL_CUDA_CALL(Memset(d_pointer,0,parent->elementCount*sizeOf(parent->type)));
    }
  }
  

  
  void DeviceBuffer::DeviceDataForCopyableData::executeResize() 
  {
    SetActiveGPU forLifeTime(device);
    
    if (d_pointer) {
      OWL_CUDA_CALL(Free(d_pointer)); d_pointer = nullptr;
    }

    if (parent->elementCount) {
      OWL_CUDA_CALL(Malloc(&d_pointer,parent->elementCount*sizeOf(parent->type)));
    }
  }
  
  void DeviceBuffer::DeviceDataForCopyableData::uploadAsync(const void *hostDataPtr, size_t offset, int64_t count)
  {
    SetActiveGPU forLifeTime(device);
    
    OWL_CUDA_CALL(MemcpyAsync((char*)d_pointer + offset,hostDataPtr,
                          ((count == -1) ? parent->elementCount : count)*sizeOf(parent->type),
                          cudaMemcpyDefault,
                          device->getStream()));
  }
  

  

  // ------------------------------------------------------------------
  // Host Pinned Buffer
  // ------------------------------------------------------------------
  
  HostPinnedBuffer::HostPinnedBuffer(Context *const context,
                                     OWLDataType type)
    : Buffer(context,type)
  {}

  /*! destructor that frees any allocated host-pinned memory */
  HostPinnedBuffer::~HostPinnedBuffer()
  {
    if (cudaHostPinnedMem) {
      OWL_CUDA_CALL_NOTHROW(FreeHost(cudaHostPinnedMem));
      cudaHostPinnedMem = nullptr;
    }
  }
  
  
  /*! pretty-printer, for debugging */
  std::string HostPinnedBuffer::toString() const
  {
    return "HostPinnedBuffer";
  }

  void HostPinnedBuffer::resize(size_t newElementCount)
  {
    if (cudaHostPinnedMem) {
      OWL_CUDA_CALL_NOTHROW(FreeHost(cudaHostPinnedMem));
      cudaHostPinnedMem = nullptr;
    }

    elementCount = newElementCount;
    if (newElementCount > 0)
      OWL_CUDA_CALL(MallocHost((void**)&cudaHostPinnedMem, sizeInBytes()));

    for (auto device : context->getDevices()) {
      getDD(device).d_pointer = cudaHostPinnedMem;
    }
  }
  
  void HostPinnedBuffer::clear()
  {
    assert(cudaHostPinnedMem);
    memset((char*)cudaHostPinnedMem, 0, sizeInBytes());
  }
  
  void HostPinnedBuffer::upload(const void *sourcePtr, size_t offset, int64_t count)
  {
    assert(cudaHostPinnedMem);
    memcpy((char*)cudaHostPinnedMem + offset, sourcePtr, (count == -1) ? sizeInBytes() : count * sizeOf(type));
  }
  
  void HostPinnedBuffer::upload(const int deviceID, const void *hostPtr, size_t offset, int64_t count)
  {
    OWL_RAISE("uploading to specific device doesn't "
              "make sense for host pinned buffers");
  }
  
  // ------------------------------------------------------------------
  // Managed Mem Buffer
  // ------------------------------------------------------------------
  
  ManagedMemoryBuffer::ManagedMemoryBuffer(Context *const context,
                                           OWLDataType type)
    : Buffer(context,type)
  {}

    /*! destructor that frees any left-over allocated memory */
  ManagedMemoryBuffer::~ManagedMemoryBuffer()
  {
    if (cudaManagedMem) {
      OWL_CUDA_CALL_NOTHROW(Free(cudaManagedMem));
      cudaManagedMem = 0;
    }
  }

  /*! pretty-printer, for debugging */
  std::string ManagedMemoryBuffer::toString() const
  {
    return "ManagedMemoryBuffer";
  }

  void ManagedMemoryBuffer::resize(size_t newElementCount)
  {
    if (cudaManagedMem) {
      OWL_CUDA_CALL_NOTHROW(Free(cudaManagedMem));
      cudaManagedMem = 0;
    }
    
    elementCount = newElementCount;
    if (newElementCount > 0) {
      OWL_CUDA_CALL(MallocManaged((void**)&cudaManagedMem, sizeInBytes()));
      unsigned char *mem_end = (unsigned char *)cudaManagedMem + sizeInBytes();
      size_t pageSize = 16*1024*1024;
      int pageID = 0;
      for (unsigned char *begin = (unsigned char *)cudaManagedMem;
           begin < mem_end;
           begin += pageSize)
        {
          unsigned char *end = std::min(begin+pageSize,mem_end);
          int devID = pageID++ % context->deviceCount();
          int cudaDevID = context->getDevice(devID)->getCudaDeviceID();
          int result = 0;
          cudaDeviceGetAttribute(&result, cudaDevAttrConcurrentManagedAccess, cudaDevID);
          if (result) {
            cudaError_t rc = cudaMemAdvise((void*)begin, end-begin,
                                           cudaMemAdviseSetPreferredLocation, cudaDevID);
            if (rc != cudaSuccess) {
#ifndef NDEBUG
              static bool alreadyWarned = false;
              if (!alreadyWarned) {
                std::cout << OWL_TERMINAL_RED
                          << "#owl: Warning - error in trying to memadvise a managed "
                          << "memory buffer: " << cudaGetErrorString(rc)
                          << " (should be OK, ignoring this)."
                          << OWL_TERMINAL_DEFAULT << std::endl;
                alreadyWarned = true;
              }
#endif
              /* clear this error */cudaGetLastError();
            }
          
          }
        }
    }
    
    for (auto device : context->getDevices())
      getDD(device).d_pointer = cudaManagedMem;
  }
  
  void ManagedMemoryBuffer::clear()
  {
    assert(cudaManagedMem);
    OWL_CUDA_CALL(Memset((char*)cudaManagedMem, 0, sizeInBytes()));
  }
  
  void ManagedMemoryBuffer::upload(const void *hostPtr, size_t offset, int64_t count)
  {
    assert(cudaManagedMem);
    cudaMemcpy((char*)cudaManagedMem + offset, hostPtr,
               (count == -1) ? sizeInBytes() : count * sizeOf(type), cudaMemcpyDefault);
  }
  
  void ManagedMemoryBuffer::upload(const int deviceID,
                                   const void *hostPtr, size_t offset, int64_t count)
  {
    OWL_RAISE("copying to a specific device doesn't"
              " make sense for a managed mem buffer");
  }

  // ------------------------------------------------------------------
  // Graphics Resource Buffer
  // ------------------------------------------------------------------
  
  /*! pretty-printer, for debugging */
  std::string GraphicsBuffer::toString() const
  {
    return "GraphicsBuffer";
  }

  GraphicsBuffer::GraphicsBuffer(Context *const context,
                                 OWLDataType type,
                                 cudaGraphicsResource_t resource)
    : Buffer(context, type)
  {}

  void GraphicsBuffer::clear()
  {
    throw std::runtime_error("graphics buffers are not a valid target for owlBufferClear()");
  }
  
  void GraphicsBuffer::resize(size_t newElementCount)
  {
    elementCount = newElementCount;
  }

  void GraphicsBuffer::upload(const void *hostPtr, size_t offset, int64_t count)
  {
    OWL_RAISE("Buffer::upload doesn' tmake sense for graphics buffers");
  }
  
  void GraphicsBuffer::upload(const int deviceID, const void *hostPtr, size_t offset, int64_t count) 
  {
    OWL_RAISE("Buffer::upload doesn' tmake sense for graphics buffers");
  }
  
  void GraphicsBuffer::map(const int deviceID, CUstream stream)
  {
    DeviceContext::SP device = context->getDevice(deviceID);
    DeviceData &dd = getDD(device);
    OWL_CUDA_CHECK(cudaGraphicsMapResources(1, &resource, stream));
    size_t size = 0;
    OWL_CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(&dd.d_pointer, &size, resource));
  }

  void GraphicsBuffer::unmap(const int deviceID, CUstream stream)
  {
    DeviceContext::SP device = context->getDevice(deviceID);
    DeviceData &dd = getDD(device);
    OWL_CUDA_CHECK(cudaGraphicsUnmapResources(1, &resource, stream));
    dd.d_pointer = nullptr;
  }

} // ::owl
