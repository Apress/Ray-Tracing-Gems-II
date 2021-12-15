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

#include "RegisteredObject.h"
#include "Texture.h"

namespace owl {

  struct Group;
  
  /*! base class for any sort of buffer type - pinned, device, managed, ... */
  struct Buffer : public RegisteredObject
  {
    typedef std::shared_ptr<Buffer> SP;
    
    /*! any device-specific data, such as optix handles, cuda device
      pointers, etc */
    struct DeviceData : public RegisteredObject::DeviceData {
      typedef std::shared_ptr<DeviceData> SP;

      /*! constructor */
      DeviceData(const DeviceContext::SP &device);

      /*! device-side pointer - depending on buffer type this may or
          may not be accessible on the host */
      void *d_pointer { 0 };
    };

    /*! construct a new buffer of given type - actual implementation
        of 'construct' done in derives class */
    Buffer(Context *const context, OWLDataType type);
    
    /*! destructor - free device data, de-regsiter, and destruct */
    virtual ~Buffer();
    
    /*! pretty-printer, for printf-debugging */
    std::string toString() const override;

    /*! get reference to given device-specific data for this object */
    inline Buffer::DeviceData &getDD(const DeviceContext::SP &device) const;

    /*! get device pointer for given buffer */
    inline const void *getPointer(const DeviceContext::SP &device) const;

    /*! return *number* of elements - number of bytes will depend on data type */
    inline size_t getElementCount() const;

    /*! return the number of *bytes* in this buffer (coiuted in size
        of device types) */
    inline size_t sizeInBytes() const { return elementCount * sizeOf(type); }
    
    /*! clear the buffer by setting its contents to zero */
    virtual void clear() = 0;
    
    /*! resize buffer to new num elements */
    virtual void resize(size_t newElementCount) = 0;
    
    /*! upload data from host, using as many bytes as required by
        elemnetCount and dataSize */
    virtual void upload(const void *hostPtr, size_t offset, int64_t count) = 0;

    /*! upload data from host, to only given device ID */
    virtual void upload(const int deviceID, const void *hostPtr, size_t offset, int64_t count) = 0;

    /*! creates the device-specific data for this group */
    RegisteredObject::DeviceData::SP createOn(const DeviceContext::SP &device) override;

    /*! destroy whatever resouces this buffer may own on the device,
        but do NOT destroy this class itself. baiscally that's a
        resize(0), but that buffer can not - and shoult not - be used
        any more after that */
    void destroy();

    /*! data type of elements contained in this buffer */
    const OWLDataType type;

    /*! number of elements */
    size_t      elementCount { 0 };
  };



  /*! a device-side buffer that has its own cuda-malloc'ed memory on
      each device. DeviceBuffers are fastest to access on the device,
      *BUT* are not visible on the host, and are replicated in the
      sense that changing to a device buffer on one GPU will not be
      visible on other GPUs */
  struct DeviceBuffer : public Buffer {
    typedef std::shared_ptr<DeviceBuffer> SP;

    /*! any device-specific data, such as optix handles, cuda device
        pointers, etc. this is a virtual base class whose derived
        classes will do the actual work */
    struct DeviceData : public Buffer::DeviceData {
      typedef std::shared_ptr<DeviceData> SP;

      /*! any device-specific data, such as optix handles, cuda device
        pointers, etc */
      DeviceData(DeviceBuffer *parent, const DeviceContext::SP &device);

      /*! destructor that releases any still-alloced memory */
      virtual ~DeviceData();

      /*! executes the resize on the given device, including freeing
          old memory, and allocating required elemnts in device
          format, as required */
      virtual void executeResize() = 0;
      
      /*! create an async upload for data from the given host data
          pointer, using the given device's cuda stream, and doing any
          required translation from host-side data (eg, a texture
          object handle) to device-side data (ie, the
          cudaTextreObject_t for that device). this will *not* wait
          for the upload to complete, so an explicit cuda sync has to
          be done to ensure no race conditiosn will occur */
      virtual void uploadAsync(const void *hostDataPtr, size_t offset, int64_t count) = 0;
      
      /*! clear the buffer by setting its contents to zero */
      virtual void clear() = 0;
    
      DeviceBuffer *const parent;
    };

    
    /*! device-data for a device buffer like other device buffers, but
        containing texture objects - ie, these are Texture::SP on the
        host, but get translated to cudaTextureObject_t's upon
        upload */
    struct DeviceDataForTextures : public DeviceData {
      DeviceDataForTextures(DeviceBuffer *parent, const DeviceContext::SP &device)
        : DeviceData(parent,device)
      {}
      void executeResize() override;
      void uploadAsync(const void *hostDataPtr, size_t offset, int64_t count) override;
      
      /*! clear the buffer by setting its contents to zero */
      void clear() override;
    
      /*! this is used only for buffers over object types (bufers of
        textures, or buffers of buffers). For those buffers, we use this
        vector to store host-side handles of the objects in this buffer,
        to ensure proper recounting */
      std::vector<Texture::SP> hostHandles;
    };



    /*! device-data for a device buffer like other device buffers, but
        containing buffers - ie, these are Buffer::SP on the host, but
        get translated to buffer descriptors upon upload */
    struct DeviceDataForBuffers : public DeviceData {
      DeviceDataForBuffers(DeviceBuffer *parent, const DeviceContext::SP &device)
        : DeviceData(parent,device)
      {}
      
      void executeResize() override;
      void uploadAsync(const void *hostDataPtr, size_t offset, int64_t count) override;
      
      /*! clear the buffer by setting its contents to zero */
      void clear() override;
    
      /*! this is used only for buffers over object types (bufers of
        textures, or buffers of buffers). For those buffers, we use this
        vector to store host-side handles of the objects in this buffer,
        to ensure proper recounting */
      std::vector<Buffer::SP> hostHandles;
    };


    /*! device-data for a device buffer like other device buffers, but
        containing buffers - ie, these are Buffer::SP on the host, but
        get translated to buffer descriptors upon upload */
    struct DeviceDataForGroups : public DeviceData {
      DeviceDataForGroups(DeviceBuffer *parent, const DeviceContext::SP &device)
        : DeviceData(parent,device)
      {}
      
      void executeResize() override;
      void uploadAsync(const void *hostDataPtr, size_t offset, int64_t count) override;
      
      /*! clear the buffer by setting its contents to zero */
      void clear() override;
      
      /*! this is used only for buffers over object types (bufers of
        textures, or buffers of buffers). For those buffers, we use this
        vector to store host-side handles of the objects in this buffer,
        to ensure proper recounting */
      std::vector<std::shared_ptr<Group>> hostHandles;
    };


    
    /*! device-data for a device buffer that contains raw, copyable
        data (float, vec3f, etc) */
    struct DeviceDataForCopyableData : public DeviceData {
      DeviceDataForCopyableData(DeviceBuffer *parent, const DeviceContext::SP &device)
        : DeviceData(parent,device)
      {}
      void executeResize() override;
      void uploadAsync(const void *hostDataPtr, size_t offset, int64_t count) override;
      
      /*! clear the buffer by setting its contents to zero */
      void clear() override;
    };

    /*! contructor - creates the right device data type based on content type */
    DeviceBuffer(Context *const context,
                 OWLDataType type);

    /*! pretty-printer, for debugging */
    std::string toString() const override;

    /*! get reference to given device-specific data for this object */
    inline DeviceData &getDD(const DeviceContext::SP &device) const;

    /*! resize this buffer - actual work will get done in DeviceData */
    void resize(size_t newElementCount) override;
    /*! upload to device data(s) of that buffer - actual work will get done in DeviceData */
    void upload(const void *hostPtr, size_t offset, int64_t count) override;
    
    /*! upload to only ONE device - only makes sense for device buffers */
    void upload(const int deviceID, const void *hostPtr, size_t offset, int64_t count) override;
      
    /*! clear the buffer by setting its contents to zero */
    void clear() override;
    
    /*! creates the device-specific data for this group */
    RegisteredObject::DeviceData::SP createOn(const DeviceContext::SP &device) override;
  };


  /*! a buffer that uses CUDA host-pinned memory. only makes sense for
      copyable data types */
  struct HostPinnedBuffer : public Buffer {
    typedef std::shared_ptr<HostPinnedBuffer> SP;
    
    HostPinnedBuffer(Context *const context,
                     OWLDataType type);
    
    /*! destructor that frees any allocated host-pinned memory */
    virtual ~HostPinnedBuffer();

    /*! pretty-printer, for debugging */
    std::string toString() const override;

    void resize(size_t newElementCount) override;
    void upload(const void *hostPtr, size_t offset, int64_t count) override;
    void upload(const int deviceID, const void *hostPtr, size_t offset, int64_t count) override;
      
    /*! clear the buffer by setting its contents to zero */
    void clear() override;

    /*! pointer to the (shared) cuda pinned mem - this gets alloced
        once and is valid on both host and devices */
    void *cudaHostPinnedMem { 0 };
  };


  
  /*! a buffer that uses CUDA 'managed' memory. only makes sense for
      copyable data types. Make sure to read up on how managed mem
      works */
  struct ManagedMemoryBuffer : public Buffer {
    typedef std::shared_ptr<ManagedMemoryBuffer> SP;
    
    ManagedMemoryBuffer(Context *const context,
                        OWLDataType type);
    
    /*! destructor that frees any left-over allocated memory */
    virtual ~ManagedMemoryBuffer();

    void resize(size_t newElementCount) override;
    void upload(const void *hostPtr, size_t offset, int64_t count) override;
    void upload(const int deviceID, const void *hostPtr, size_t offset, int64_t count) override;

    /*! pretty-printer, for debugging */
    std::string toString() const override;
      
    /*! clear the buffer by setting its contents to zero */
    void clear() override;

    /*! pointer to the (shared) cuda managed mem - this gets alloced
        once and is valid on both host and devices */
    void *cudaManagedMem { 0 };
  };


  /*! a special graphics resource buffer that, upon mapping, will map
      that graphics resource */
  struct GraphicsBuffer : public Buffer {
    typedef std::shared_ptr<GraphicsBuffer> SP;

    GraphicsBuffer(Context* const context,
                   OWLDataType type,
                   cudaGraphicsResource_t resource);

    void map(const int deviceID=0, CUstream stream=0);
    void unmap(const int deviceID=0, CUstream stream=0);

    void resize(size_t newElementCount) override;
    void upload(const void *hostPtr, size_t offset, int64_t count) override;
    void upload(const int deviceID, const void *hostPtr, size_t offset, int64_t count) override;
      
    /*! clear the buffer by setting its contents to zero */
    void clear() override;

    /*! the cuda graphics resource to map to - note that this is
        probably valid on only one GPU */
    cudaGraphicsResource_t resource;
    
    /*! pretty-printer, for debugging */
    std::string toString() const override;
  };



  // ------------------------------------------------------------------
  // implementation section
  // ------------------------------------------------------------------
  
  
  /*! return *number* of elements - number of bytes will depend on data type */
  inline size_t Buffer::getElementCount() const
  {
    return elementCount;
  }
  
  /*! get device pointer for given buffer */
  const void *Buffer::getPointer(const DeviceContext::SP &device) const
  {
    assert(device);
    return getDD(device).d_pointer;
  }
  
  /*! get reference to given device-specific data for this object */
  inline Buffer::DeviceData &Buffer::getDD(const DeviceContext::SP &device) const
  {
    assert(device && device->ID >= 0 && device->ID < (int)deviceData.size());
    return deviceData[device->ID]->as<Buffer::DeviceData>();
  }
  
    /*! get reference to given device-specific data for this object */
  inline DeviceBuffer::DeviceData &DeviceBuffer::getDD(const DeviceContext::SP &device) const
  {
    assert(device && device->ID >= 0 && device->ID < (int)deviceData.size());
    return deviceData[device->ID]->as<DeviceData>();
  }

} // ::owl
