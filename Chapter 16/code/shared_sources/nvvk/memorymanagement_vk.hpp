/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <assert.h>
#include <platform.h>
#include <string>
#include <vector>

#include <nvh/trangeallocator.hpp>
#include <vulkan/vulkan_beta.h>
#include <vulkan/vulkan_core.h>

namespace nvvk {

// safe-ish value for most desktop hw http://vulkan.gpuinfo.org/displayextensionproperty.php?name=maxMemoryAllocationSize
#define NVVK_DEFAULT_MAX_MEMORY_ALLOCATIONSIZE (VkDeviceSize(2 * 1024) * 1024 * 1024)

#define NVVK_DEFAULT_MEMORY_BLOCKSIZE (VkDeviceSize(128) * 1024 * 1024)

#define NVVK_DEFAULT_STAGING_BLOCKSIZE (VkDeviceSize(64) * 1024 * 1024)

//////////////////////////////////////////////////////////////////////////
/**
  This framework assumes that memory heaps exists that support:

  - VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    for uploading data to the device
  - VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT & VK_MEMORY_PROPERTY_HOST_CACHED_BIT
    for downloading data from the device

  This is typical on all major desktop platforms and vendors.
  See http://vulkan.gpuinfo.org for information of various devices and platforms.

  # functions in nvvk

  * getMemoryInfo : fills the VkMemoryAllocateInfo based on device's memory properties and memory requirements and property flags. Returns `true` on success.
*/

// returns true on success
bool getMemoryInfo(const VkPhysicalDeviceMemoryProperties& memoryProperties,
                   const VkMemoryRequirements&             memReqs,
                   VkMemoryPropertyFlags                   properties,
                   VkMemoryAllocateInfo&                   memInfo,
                   bool preferDevice = true);  // special case if zero properties are unsupported, otherwise use host

//////////////////////////////////////////////////////////////////////////

static const uint32_t INVALID_ID_INDEX = ~0;

struct Allocation
{
  VkDeviceMemory mem    = VK_NULL_HANDLE;
  VkDeviceSize   offset = 0;
  VkDeviceSize   size   = 0;
};

class AllocationID
{
  friend class DeviceMemoryAllocator;

private:
  uint32_t index      = INVALID_ID_INDEX;
  uint32_t generation = 0;

  void     invalidate() { index = INVALID_ID_INDEX; }
  uint32_t instantiate(uint32_t newIndex)
  {
    uint32_t oldIndex = index;
    index             = newIndex;
    generation++;

    return oldIndex;
  }

public:
  bool isValid() const { return index != INVALID_ID_INDEX; }
  bool isEqual(const AllocationID& other) const { return index == other.index && generation == other.generation; }

  operator bool() const { return isValid(); }

  friend bool operator==(const AllocationID& lhs, const AllocationID& rhs) { return rhs.isEqual(lhs); }
};


//////////////////////////////////////////////////////////////////////////
/**
  # class nvvk::DeviceMemoryAllocator

  DeviceMemoryAllocator allocates and manages device memory in fixed-size memory blocks.

  It sub-allocates from the blocks, and can re-use memory if it finds empty
  regions. Because of the fixed-block usage, you can directly create resources
  and don't need a phase to compute the allocation sizes first.

  It will create compatible chunks according to the memory requirements and
  usage flags. Therefore you can easily create mappable host allocations
  and delete them after usage, without inferring device-side allocations.

  We return `AllocationID` rather than the allocation details directly, which
  you can query separately.

  Several utility functions are provided to handle the binding of memory
  directly with the resource creation of buffers, images and acceleration
  structures. This utilities also make implicit use of Vulkan's dedicated
  allocation mechanism.

  > **WARNING** : The memory manager serves as proof of concept for some key concepts
  > however it is not meant for production use and it currently lacks de-fragmentation logic
  > as well. You may want to look at [VMA](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator)
  > for a more production-focused solution.

  You can derive from this calls and overload the 

  Example :
  ~~~ C++
  nvvk::DeviceMemoryAllocator memAllocator;

  memAllocator.init(device, physicalDevice);

  // low-level
  aid = memAllocator.alloc(memRequirements,...);
  ...
  memAllocator.free(aid);

  // utility wrapper
  buffer = memAllocator.createBuffer(bufferSize, bufferUsage, bufferAid);
  ...
  memAllocator.free(bufferAid);


  // It is also possible to not track individual resources
  // and free everything in one go. However, this is
  // not recommended for general purpose use.

  bufferA = memAllocator.createBuffer(sizeA, usageA);
  bufferB = memAllocator.createBuffer(sizeB, usageB);
  ...
  memAllocator.freeAll();

  ~~~
*/
class DeviceMemoryAllocator
{

public:
  static const float DEFAULT_PRIORITY;

  DeviceMemoryAllocator(DeviceMemoryAllocator const&) = delete;
  DeviceMemoryAllocator& operator=(DeviceMemoryAllocator const&) = delete;


  virtual ~DeviceMemoryAllocator()
  {
#ifndef NDEBUG
    // If all memory was released properly, no blocks should be alive at this point
    assert(m_blocks.empty());
#endif
    deinit();
  }


  // system related

  DeviceMemoryAllocator() { m_debugName = "nvvk::DeviceMemoryAllocator:" + std::to_string((uint64_t)this); }
  DeviceMemoryAllocator(VkDevice         device,
                        VkPhysicalDevice physicalDevice,
                        VkDeviceSize     blockSize = NVVK_DEFAULT_MEMORY_BLOCKSIZE,
                        VkDeviceSize     maxSize   = NVVK_DEFAULT_MAX_MEMORY_ALLOCATIONSIZE)
  {
    init(device, physicalDevice, blockSize, maxSize);
  }

  void init(VkDevice         device,
            VkPhysicalDevice physicalDevice,
            VkDeviceSize     blockSize = NVVK_DEFAULT_MEMORY_BLOCKSIZE,
            VkDeviceSize     maxSize   = NVVK_DEFAULT_MAX_MEMORY_ALLOCATIONSIZE);

  void setDebugName(const std::string& name) { m_debugName = name; }

  // requires VK_EXT_memory_priority, default is false
  void setPrioritySupported(bool state) { m_supportsPriority = state; }

  // frees all blocks independent of individual allocations
  // use only if you know the lifetime of all resources from this allocator.
  void freeAll();

  // asserts on all resources being freed properly
  void deinit();

  // get utilization of block allocations
  float getUtilization(VkDeviceSize& allocatedSize, VkDeviceSize& usedSize) const;
  // get total amount of active blocks / VkDeviceMemory allocations
  uint32_t getActiveBlockCount() const { return m_activeBlockCount; }

  // dump detailed stats via nvprintfLevel(LOGLEVEL_INFO
  void nvprintReport() const;

  void getTypeStats(uint32_t     count[VK_MAX_MEMORY_TYPES],
                    VkDeviceSize used[VK_MAX_MEMORY_TYPES],
                    VkDeviceSize allocated[VK_MAX_MEMORY_TYPES]) const;

  VkDevice                                getDevice() const;
  VkPhysicalDevice                        getPhysicalDevice() const;
  const VkPhysicalDeviceMemoryProperties& getMemoryProperties() const;
  VkDeviceSize                            getMaxAllocationSize() const;

  //////////////////////////////////////////////////////////////////////////

  // subsequent allocations (and creates) will use the provided priority
  // ignored if setPrioritySupported is not enabled
  float setPriority(float priority = DEFAULT_PRIORITY)
  {
    float old  = m_priority;
    m_priority = priority;
    return old;
  }

  float getPriority() const { return m_priority; }

  // subsequent allocations (and creates) will use the provided flags
  void setAllocateFlags(VkMemoryAllocateFlags flags, bool enabled)
  {
    if(enabled)
    {
      m_allocateFlags |= flags;
    }
    else
    {
      m_allocateFlags &= ~flags;
    }
  }

  void setAllocateDeviceMask(uint32_t allocateDeviceMask, bool enabled)
  {
    if(enabled)
    {
      m_allocateDeviceMask |= allocateDeviceMask;
    }
    else
    {
      m_allocateDeviceMask &= ~allocateDeviceMask;
    }
  }

  VkMemoryAllocateFlags getAllocateFlags() const { return m_allocateFlags; }
  uint32_t              getAllocateDeviceMask() const { return m_allocateDeviceMask; }

  // make individual raw allocations.
  // there is also utilities that combine creation of buffers/images etc. with binding
  // the memory below.
  AllocationID alloc(const VkMemoryRequirements& memReqs,
                     VkMemoryPropertyFlags       memProps,
                     bool                        isLinear,  // buffers are linear, optimal tiling textures are not
                     const VkMemoryDedicatedAllocateInfo* dedicated,
                     VkResult&                            result)
  {
    return allocInternal(memReqs, memProps, isLinear, dedicated, result, true);
  }

  AllocationID alloc(const VkMemoryRequirements& memReqs,
                     VkMemoryPropertyFlags       memProps = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                     bool isLinear = true,  // buffers are linear, optimal tiling textures are not
                     const VkMemoryDedicatedAllocateInfo* dedicated = nullptr)
  {
    VkResult result;
    return allocInternal(memReqs, memProps, isLinear, dedicated, result, true);
  }

  // unless you use the freeAll mechanism, each allocation must be freed individually
  void free(AllocationID allocationID);

  // returns the detailed information from an allocationID
  const Allocation& getAllocation(AllocationID id) const;

  // can have multiple map/unmaps at once, but must be paired
  // internally will keep the vk mapping active as long as one map is active
  void* map(AllocationID allocationID);
  void  unmap(AllocationID allocationID);

  template <class T>
  T* mapT(AllocationID allocationID)
  {
    return (T*)map(allocationID);
  }

  //////////////////////////////////////////////////////////////////////////

  // utility functions to create resources and bind their memory directly

  // subsequent creates will use dedicated allocations (mostly for debugging purposes)
  inline void setForceDedicatedAllocation(bool state) { m_forceDedicatedAllocation = state; }
  // subsequent createBuffers will also use these flags
  inline void setDefaultBufferUsageFlags(VkBufferUsageFlags usage) { m_defaultBufferUsageFlags = usage; }

  VkImage createImage(const VkImageCreateInfo& createInfo, AllocationID& allocationID, VkMemoryPropertyFlags memProps, VkResult& result);
  VkImage createImage(const VkImageCreateInfo& createInfo, AllocationID& allocationID, VkMemoryPropertyFlags memProps = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
  {
    VkResult result;
    return createImage(createInfo, allocationID, memProps, result);
  }
  VkImage createImage(const VkImageCreateInfo& createInfo, VkMemoryPropertyFlags memProps, VkResult& result)
  {
    AllocationID id;
    return createImage(createInfo, id, memProps, result);
  }
  VkImage createImage(const VkImageCreateInfo& createInfo, VkMemoryPropertyFlags memProps = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
  {
    AllocationID id;
    return createImage(createInfo, id, memProps);
  }


  VkBuffer createBuffer(const VkBufferCreateInfo& createInfo, AllocationID& allocationID, VkMemoryPropertyFlags memProps, VkResult& result);
  VkBuffer createBuffer(const VkBufferCreateInfo& createInfo, AllocationID& allocationID, VkMemoryPropertyFlags memProps = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
  {
    VkResult result;
    return createBuffer(createInfo, allocationID, memProps, result);
  }
  VkBuffer createBuffer(const VkBufferCreateInfo& createInfo, VkMemoryPropertyFlags memProps, VkResult& result)
  {
    AllocationID id;
    return createBuffer(createInfo, id, memProps, result);
  }
  VkBuffer createBuffer(const VkBufferCreateInfo& createInfo, VkMemoryPropertyFlags memProps = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
  {
    AllocationID id;
    return createBuffer(createInfo, id, memProps);
  }

  VkBuffer createBuffer(VkDeviceSize size,
                        VkBufferUsageFlags usage,  // combined with m_defaultBufferUsageFlags and VK_BUFFER_USAGE_TRANSFER_DST_BIT
                        AllocationID&         allocationID,
                        VkMemoryPropertyFlags memProps,
                        VkResult&             result);
  VkBuffer createBuffer(VkDeviceSize size,
                        VkBufferUsageFlags usage,  // combined with m_defaultBufferUsageFlags and VK_BUFFER_USAGE_TRANSFER_DST_BIT
                        AllocationID&         allocationID,
                        VkMemoryPropertyFlags memProps = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
  {
    VkResult result;
    return createBuffer(size, usage, allocationID, memProps, result);
  }

#if VK_NV_ray_tracing
  VkAccelerationStructureNV createAccStructure(const VkAccelerationStructureCreateInfoNV& createInfo,
                                               AllocationID&                              allocationID,
                                               VkMemoryPropertyFlags                      memProps,
                                               VkResult&                                  result);
  VkAccelerationStructureNV createAccStructure(const VkAccelerationStructureCreateInfoNV& createInfo,
                                               AllocationID&                              allocationID,
                                               VkMemoryPropertyFlags memProps = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
  {
    VkResult result;
    return createAccStructure(createInfo, allocationID, memProps, result);
  }
#endif


protected:
  static const VkMemoryDedicatedAllocateInfo* DEDICATED_PROXY;
  static int                                  s_allocDebugBias;

  struct BlockID
  {
    uint32_t index      = INVALID_ID_INDEX;
    uint32_t generation = 0;

    bool     isEqual(const BlockID& other) const { return index == other.index && generation == other.generation; }
    uint32_t instantiate(uint32_t newIndex)
    {
      uint32_t oldIndex = index;
      index             = newIndex;
      generation++;

      return oldIndex;
    }

    friend bool operator==(const BlockID& lhs, const BlockID& rhs) { return rhs.isEqual(lhs); }
  };

  struct Block
  {
    BlockID                   id;  // index to self, or next free item
    VkDeviceMemory            mem = VK_NULL_HANDLE;
    nvh::TRangeAllocator<256> range;

    VkDeviceSize allocationSize;
    VkDeviceSize usedSize;

    // to avoid management of pages via limits::bufferImageGranularity,
    // a memory block is either fully linear, or non-linear
    bool                  isLinear;
    bool                  isDedicated;
    bool                  isFirst;  // first memory block of a type
    float                 priority;
    VkMemoryAllocateFlags allocateFlags;
    uint32_t              allocateDeviceMask;

    uint32_t memoryTypeIndex;
    uint32_t allocationCount;
    uint32_t mapCount;
    uint32_t mappable;
    uint8_t* mapped;

    Block& operator=(Block&&) = default;
    Block(Block&&)            = default;
    Block(const Block&)       = default;
    Block()                   = default;
  };

  struct AllocationInfo
  {
    AllocationID id;  // index to self, or next free item
    Allocation   allocation;
    uint32_t     blockOffset;
    uint32_t     blockSize;
    BlockID      block;
  };

  VkDevice     m_device            = VK_NULL_HANDLE;
  VkDeviceSize m_blockSize         = 0;
  VkDeviceSize m_allocatedSize     = 0;
  VkDeviceSize m_usedSize          = 0;
  VkDeviceSize m_maxAllocationSize = NVVK_DEFAULT_MAX_MEMORY_ALLOCATIONSIZE;

  std::vector<Block>          m_blocks;
  std::vector<AllocationInfo> m_allocations;

  // linked-list to next free allocation
  uint32_t m_freeAllocationIndex = INVALID_ID_INDEX;
  // linked-list to next free block
  uint32_t m_freeBlockIndex   = INVALID_ID_INDEX;
  uint32_t m_activeBlockCount = 0;

  VkPhysicalDeviceMemoryProperties m_memoryProperties;
  VkPhysicalDevice                 m_physicalDevice = VK_NULL_HANDLE;

  float                 m_priority           = DEFAULT_PRIORITY;
  VkMemoryAllocateFlags m_allocateFlags      = 0;
  uint32_t              m_allocateDeviceMask = 0;

  VkBufferUsageFlags m_defaultBufferUsageFlags  = 0;
  bool               m_forceDedicatedAllocation = false;
  bool               m_supportsPriority         = false;
  // heuristic that doesn't immediately free the first memory block of a specific memorytype
  bool m_keepFirst = true;

  std::string m_debugName;

  AllocationID allocInternal(const VkMemoryRequirements& memReqs,
                             VkMemoryPropertyFlags       memProps,
                             bool isLinear,  // buffers are linear, optimal tiling textures are not
                             const VkMemoryDedicatedAllocateInfo* dedicated,
                             VkResult&                            result,
                             bool                                 preferDevice);

  AllocationID createID(Allocation& allocation, BlockID block, uint32_t blockOffset, uint32_t blockSize);
  void         destroyID(AllocationID id);

  const AllocationInfo& getInfo(AllocationID id) const
  {
    assert(m_allocations[id.index].id.isEqual(id));

    return m_allocations[id.index];
  }

  Block& getBlock(BlockID id)
  {
    Block& block = m_blocks[id.index];
    assert(block.id.isEqual(id));
    return block;
  }

  //////////////////////////////////////////////////////////////////////////
  // For derived memory allocators you can do special purpose operations via overloading these functions.
  // A typical use-case would be export/import the memory to another API.

  virtual VkResult allocBlockMemory(BlockID id, VkMemoryAllocateInfo& memInfo, VkDeviceMemory& deviceMemory)
  {
    //s_allocDebugBias++;
    return vkAllocateMemory(m_device, &memInfo, nullptr, &deviceMemory);
  }
  virtual void freeBlockMemory(BlockID id, VkDeviceMemory deviceMemory)
  {
    //s_allocDebugBias--;
    vkFreeMemory(m_device, deviceMemory, nullptr);
  }
  virtual void resizeBlocks(uint32_t count) {}

  virtual VkResult createBufferInternal(VkDevice device, const VkBufferCreateInfo* info, VkBuffer* buffer)
  {
    return vkCreateBuffer(device, info, nullptr, buffer);
  }

  virtual VkResult createImageInternal(VkDevice device, const VkImageCreateInfo* info, VkImage* image)
  {
    return vkCreateImage(device, info, nullptr, image);
  }
};

//////////////////////////////////////////////////////////////////
/**
  # class nvvk::StagingMemoryManager

  StagingMemoryManager class is a utility that manages host visible
  buffers and their allocations in an opaque fashion to assist
  asynchronous transfers between device and host.

  The collection of the transfer resources is represented by nvvk::StagingID.

  The necessary buffer space is sub-allocated and recycled in blocks internally.
  This way we avoid creating lots of small VkBuffers and avoid calling the Vulkan
  API at all. While Vulkan is more efficient than previous APIs, creating lots
  of objects for it, is still not good for overall performance. It will result 
  into more cache misses and use more system memory over all.

  The default implementation will create one dedicated memory allocation per block.
  You can derive from this class and overload the virtual functions,
  if you want to use a different allocation system.

  - **allocBlockMemory**
  - **freeBlockMemory**

  > **WARNING:**
  > - cannot manage a copy > 4 GB

  Usage:
  - Enqueue transfers into your VkCommandBuffer and then finalize the copy operations.
  - Associate the copy operations with a VkFence
  - The release of the resources allows to safely recycle the buffer space for future transfers.
  
  > We use fences as a way to garbage collect here, however a more robust solution
  > may be implementing some sort of ticketing/timeline system.
  > If a fence is recycled, then this class may not be aware that the fence represents a different
  > submission, likewise if the fence is deleted elsewhere problems can occur.

  Example :

  ~~~ C++
  StagingMemoryManager  staging;
  staging.init(device, physicalDevice);


  // Enqueue copy operations of data to target buffer.
  // This internally manages the required staging resources
  staging.cmdToBuffer(cmd, targetBufer, 0, targetSize, targetData);

  // you can also get access to a temporary mapped pointer and fill
  // the staging buffer directly
  vertices = staging.cmdToBufferT<Vertex>(cmd, targetBufer, 0, targetSize);

  // OPTION A:
  // associate all previous copy operations with a fence
  staging.finalizeResources( fence );
  ..
  // every once in a while call
  staging.releaseResources();

  // OPTION B
  // alternatively manage the resource release yourself
  sid = staging.finalizeResources();

  ... you need to ensure these uploads completed

  staging.releaseResources();

  ~~~
*/

class StagingMemoryManager
{
public:
  //////////////////////////////////////////////////////////////////////////

  StagingMemoryManager(StagingMemoryManager const&) = delete;
  StagingMemoryManager& operator=(StagingMemoryManager const&) = delete;

  StagingMemoryManager() { m_debugName = "nvvk::StagingMemoryManager:" + std::to_string((uint64_t)this); }
  StagingMemoryManager(VkDevice device, VkPhysicalDevice physicalDevice, VkDeviceSize stagingBlockSize = NVVK_DEFAULT_STAGING_BLOCKSIZE)
  {
    init(device, physicalDevice, stagingBlockSize);
  }

  ~StagingMemoryManager() { deinit(); }

  void init(VkDevice device, VkPhysicalDevice physicalDevice, VkDeviceSize stagingBlockSize = NVVK_DEFAULT_STAGING_BLOCKSIZE);
  void deinit();
  void setDebugName(const std::string& name) { m_debugName = name; }

  // if true (default) we free the memory completely when released
  // otherwise we would keep blocks for re-use around, unless freeUnused() is called
  void setFreeUnusedOnRelease(bool state) { m_freeOnRelease = state; }

  // test if there is enough space in current allocations
  bool fitsInAllocated(VkDeviceSize size, bool toDevice = true) const;

  // if data != nullptr memcpies to mapping and returns nullptr
  // otherwise returns temporary mapping (valid until "complete" functions)
  void* cmdToImage(VkCommandBuffer                 cmd,
                   VkImage                         image,
                   const VkOffset3D&               offset,
                   const VkExtent3D&               extent,
                   const VkImageSubresourceLayers& subresource,
                   VkDeviceSize                    size,
                   const void*                     data);


  // if data != nullptr memcpies to mapping and returns nullptr
  // otherwise returns temporary mapping (valid until appropriate release)
  void* cmdToBuffer(VkCommandBuffer cmd, VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size, const void* data);

  template <class T>
  T* cmdToBufferT(VkCommandBuffer cmd, VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size)
  {
    return (T*)cmdToBuffer(cmd, buffer, offset, size, nullptr);
  }

  // pointer is only valid until associated resources haven't been released
  const void* cmdFromBuffer(VkCommandBuffer cmd, VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size);

  template <class T>
  const T* cmdFromBufferT(VkCommandBuffer cmd, VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size)
  {
    return (const T*)cmdFromBuffer(cmd, buffer, offset, size);
  }

  // FIXME readback "from" tasks may not want to use a fence

  // closes the batch of staging resources since last finalizeResources call
  // and associates it with a fence for later release.
  void finalizeResources(VkFence fence = VK_NULL_HANDLE);

  // releases the staging resources whose fences have completed
  // and those who had no fence at all
  void releaseResources();

  // frees staging memory no longer in use
  void freeUnused() { free(true); }

  float getUtilization(VkDeviceSize& allocatedSize, VkDeviceSize& usedSize) const;

protected:
  // The implementation uses two major arrays:
  // - Block stores VkBuffers that we sub-allocate the staging space from
  // - StagingSet stores all such sub-allocations that were used
  //   in one batch of operations. Each batch is closed with
  //   finalizeResources, and typically associated with a fence.
  //   As such the resources are given by for recycling if the fence completed.

  // To recycle Block and StagingSet structures within the arrays
  // we use a linked list of array indices. The "index" element
  // in the struct refers to the next free list item, or itself
  // when in use.

  struct Block
  {
    uint32_t                  index    = INVALID_ID_INDEX;
    VkDeviceSize              size     = 0;
    VkBuffer                  buffer   = VK_NULL_HANDLE;
    VkDeviceMemory            memory   = VK_NULL_HANDLE;
    bool                      toDevice = true;
    nvh::TRangeAllocator<256> range;
    uint8_t*                  mapping;
  };

  struct Entry
  {
    uint32_t block;
    uint32_t offset;
    uint32_t size;
  };

  struct StagingSet
  {
    uint32_t           index = INVALID_ID_INDEX;
    VkFence            fence = VK_NULL_HANDLE;
    std::vector<Entry> entries;
  };

  VkDevice         m_device         = VK_NULL_HANDLE;
  VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;
  uint32_t         m_memoryTypeIndex;
  VkDeviceSize     m_stagingBlockSize;
  bool             m_freeOnRelease;

  std::vector<Block>      m_blocks;
  std::vector<StagingSet> m_sets;

  // active staging Index, must be valid at all items
  uint32_t m_stagingIndex;
  // linked-list to next free staging set
  uint32_t m_freeStagingIndex;
  // linked list to next free block
  uint32_t m_freeBlockIndex;

  VkDeviceSize m_allocatedSize;
  VkDeviceSize m_usedSize;

  std::string m_debugName;

  uint32_t setIndexValue(uint32_t& index, uint32_t newValue)
  {
    uint32_t oldValue = index;
    index             = newValue;
    return oldValue;
  }

  void free(bool unusedOnly);
  void freeBlock(Block& block);

  uint32_t newStagingIndex();

  void* getStagingSpace(VkDeviceSize size, VkBuffer& buffer, VkDeviceSize& offset, bool toDevice);

  Block& getBlock(uint32_t index)
  {
    Block& block = m_blocks[index];
    assert(block.index == index);
    return block;
  }

  void releaseResources(uint32_t stagingID);

  //////////////////////////////////////////////////////////////////////////
  // You can specialize the staging buffer allocation mechanism used.
  // The default is using one dedicated VkDeviceMemory per staging VkBuffer.

  // must fill block.buffer, memory, mapping
  virtual VkResult allocBlockMemory(uint32_t id, VkDeviceSize size, bool toDevice, Block& block);
  virtual void     freeBlockMemory(uint32_t id, const Block& block);
  virtual void     resizeBlocks(uint32_t num) {}
};

//////////////////////////////////////////////////////////////////
/**
  # class nvvk::StagingMemoryManagerDma

  Derives from nvvk::StagingMemoryManager and uses the referenced nvvk::DeviceMemoryAllocator
  for allocations.

  ~~~ C++
  DeviceMemoryAllocator    memAllocator;
  memAllocator.init(device, physicalDevice);

  StagingMemoryManagerDma  staging;
  staging.init(memAllocator);

  // rest as usual
  staging.cmdToBuffer(cmd, targetBufer, 0, targetSize, targetData);
  ~~~
*/

class StagingMemoryManagerDma : public StagingMemoryManager
{
public:
  StagingMemoryManagerDma(DeviceMemoryAllocator* memAllocator, VkDeviceSize stagingBlockSize = NVVK_DEFAULT_STAGING_BLOCKSIZE)
  {
    init(memAllocator, stagingBlockSize);
  }
  StagingMemoryManagerDma() {}

  void init(DeviceMemoryAllocator* memAllocator, VkDeviceSize stagingBlockSize = NVVK_DEFAULT_STAGING_BLOCKSIZE)
  {
    StagingMemoryManager::init(memAllocator->getDevice(), memAllocator->getPhysicalDevice(), stagingBlockSize);
    m_memAllocator = memAllocator;
  }

protected:
  DeviceMemoryAllocator*    m_memAllocator;
  std::vector<AllocationID> m_blockAllocs;

  VkResult allocBlockMemory(uint32_t index, VkDeviceSize size, bool toDevice, Block& block) override;
  void     freeBlockMemory(uint32_t index, const Block& block) override;

  void resizeBlocks(uint32_t num) override;
};

}  // namespace nvvk
