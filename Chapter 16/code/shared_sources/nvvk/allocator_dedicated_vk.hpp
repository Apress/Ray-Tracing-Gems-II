/* Copyright (c) 2014-2018, NVIDIA CORPORATION. All rights reserved.
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

#include <cassert>
#include <vector>
#include <vulkan/vulkan.h>

#include "error_vk.hpp"
#include "images_vk.hpp"
#include "samplers_vk.hpp"

#include <memory>


//////////////////////////////////////////////////////////////////////////
/**
  # class nvvk::AllocatorDedicated

  This is the allocator specialization using only Vulkan where there will be one memory
  allocation for each buffer or image.
  See more details in description of [nvvk::AllocatorDma](#class-nvvkallocatordma) for the
  general use of allocator classes.

  > Note: this should be used only when really needed, as it is making one allocation per buffer,
  >       which is not efficient. 

  # Initialization

  ~~~~~~ C++
      nvvk::AllocatorVk m_alloc;
      m_alloc.init(device, physicalDevice);
  ~~~~~~ 
*/

namespace nvvk {

// Objects
struct BufferDedicated
{
  VkBuffer       buffer{VK_NULL_HANDLE};
  VkDeviceMemory allocation{VK_NULL_HANDLE};
};

struct ImageDedicated
{
  VkImage        image{VK_NULL_HANDLE};
  VkDeviceMemory allocation{VK_NULL_HANDLE};
};

struct TextureDedicated
{
  VkImage               image = VK_NULL_HANDLE;
  VkDeviceMemory        allocation{VK_NULL_HANDLE};
  VkDescriptorImageInfo descriptor{};
};

#if VK_NV_ray_tracing
struct AccelerationDedicatedNV
{
  VkAccelerationStructureNV accel{VK_NULL_HANDLE};
  VkDeviceMemory            allocation{VK_NULL_HANDLE};
};
#endif
#if VK_KHR_acceleration_structure
struct AccelerationDedicatedKHR
{
  VkAccelerationStructureKHR accel{VK_NULL_HANDLE};
  BufferDedicated            buffer;
};
#endif

//--------------------------------------------------------------------------------------------------
// Allocator for buffers, images and acceleration structure using Pure Vulkan
//
class AllocatorDedicated
{
public:
  AllocatorDedicated(AllocatorDedicated const&) = delete;
  AllocatorDedicated& operator=(AllocatorDedicated const&) = delete;

  AllocatorDedicated() = default;
  // All staging buffers must be cleared before
  ~AllocatorDedicated() { assert(m_stagingBuffers.empty()); }

  //--------------------------------------------------------------------------------------------------
  // Initialization of the allocator
  void init(VkDevice device, VkPhysicalDevice physicalDevice)
  {
    m_device         = device;
    m_physicalDevice = physicalDevice;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &m_memoryProperties);
    m_samplerPool.init(device);
  }

  void deinit() { m_samplerPool.deinit(); }

  //--------------------------------------------------------------------------------------------------
  // Basic buffer creation
  virtual BufferDedicated createBuffer(const VkBufferCreateInfo& info_, const VkMemoryPropertyFlags memUsage_ = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
  {
    BufferDedicated resultBuffer;
    // 1. Create Buffer (can be overloaded)
    CreateBufferEx(info_, &resultBuffer.buffer);  // Potentially adding handle

    // 2. Find memory requirements
    VkMemoryRequirements2           memReqs{VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};
    VkMemoryDedicatedRequirements   dedicatedRegs{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS};
    VkBufferMemoryRequirementsInfo2 bufferReqs{VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2};

    bufferReqs.buffer = resultBuffer.buffer;
    memReqs.pNext     = &dedicatedRegs;
    vkGetBufferMemoryRequirements2(m_device, &bufferReqs, &memReqs);

    // Device Address
    VkMemoryAllocateFlagsInfo memFlagInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    if(info_.usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT)
      memFlagInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;

    // 3. Allocate memory
    VkMemoryAllocateInfo memAlloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    memAlloc.allocationSize  = memReqs.memoryRequirements.size;
    memAlloc.memoryTypeIndex = getMemoryType(memReqs.memoryRequirements.memoryTypeBits, memUsage_);
    memAlloc.pNext           = &memFlagInfo;
    resultBuffer.allocation  = AllocateMemory(memAlloc);
    checkMemory(resultBuffer.allocation);

    // 4. Bind memory to buffer
    NVVK_CHECK(vkBindBufferMemory(m_device, resultBuffer.buffer, resultBuffer.allocation, 0));

    return resultBuffer;
  }

  //--------------------------------------------------------------------------------------------------
  // Simple buffer creation
  BufferDedicated createBuffer(VkDeviceSize                size_     = 0,
                               VkBufferUsageFlags          usage_    = VkBufferUsageFlags(),
                               const VkMemoryPropertyFlags memUsage_ = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
  {
    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    info.size  = size_;
    info.usage = usage_;

    return createBuffer(info, memUsage_);
  }

  //--------------------------------------------------------------------------------------------------
  // Staging buffer creation, uploading data to device buffer
  BufferDedicated createBuffer(const VkCommandBuffer& cmdBuf,
                               const VkDeviceSize&    size_,
                               const void*            data_,
                               VkBufferUsageFlags     usage_,
                               VkMemoryPropertyFlags  memProps = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
  {
    // 1. Create staging buffer
    BufferDedicated stageBuffer = createBuffer(size_, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    m_stagingBuffers.push_back(stageBuffer);  // Remember the buffers to delete

    // 2. Copy the data to memory
    if(data_)
    {
      void* mapped = nullptr;
      NVVK_CHECK(vkMapMemory(m_device, stageBuffer.allocation, 0, size_, 0, &mapped));
      memcpy(mapped, data_, size_);
      vkUnmapMemory(m_device, stageBuffer.allocation);
    }

    // 3. Create the result buffer
    VkBufferCreateInfo createInfoR{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    createInfoR.size             = size_;
    createInfoR.usage            = usage_ | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    BufferDedicated resultBuffer = createBuffer(createInfoR, memProps);

    // 4. Copy staging (need to submit command buffer, flushStaging must be done after submitting)
    VkBufferCopy region{0, 0, size_};
    vkCmdCopyBuffer(cmdBuf, stageBuffer.buffer, resultBuffer.buffer, 1, &region);
    return resultBuffer;
  }

  //--------------------------------------------------------------------------------------------------
  // Staging buffer creation, uploading data to device buffer
  template <typename T>
  BufferDedicated createBuffer(const VkCommandBuffer& cmdBuff,
                               const std::vector<T>&  data_,
                               VkBufferUsageFlags     usage_,
                               VkMemoryPropertyFlags  memProps_ = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
  {
    return createBuffer(cmdBuff, sizeof(T) * data_.size(), data_.data(), usage_, memProps_);
  }


  //--------------------------------------------------------------------------------------------------
  // Basic image creation
  ImageDedicated createImage(const VkImageCreateInfo& info_, const VkMemoryPropertyFlags memUsage_ = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
  {
    ImageDedicated resultImage;
    // 1. Create image
    CreateImageEx(info_, &resultImage.image);

    // 2. Find memory requirements
    VkMemoryRequirements2          memReqs{VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};
    VkMemoryDedicatedRequirements  dedicatedRegs{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS};
    VkImageMemoryRequirementsInfo2 imageReqs{VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2};

    imageReqs.image = resultImage.image;
    memReqs.pNext   = &dedicatedRegs;
    vkGetImageMemoryRequirements2(m_device, &imageReqs, &memReqs);


    // 3. Allocate memory
    VkMemoryAllocateInfo memAllocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    memAllocInfo.allocationSize  = memReqs.memoryRequirements.size;
    memAllocInfo.memoryTypeIndex = getMemoryType(memReqs.memoryRequirements.memoryTypeBits, memUsage_);
    resultImage.allocation       = AllocateMemory(memAllocInfo);
    checkMemory(resultImage.allocation);

    // 4. Bind memory to image
    NVVK_CHECK(vkBindImageMemory(m_device, resultImage.image, resultImage.allocation, 0));

    return resultImage;
  }


  //--------------------------------------------------------------------------------------------------
  // Create an image with data
  //
  ImageDedicated createImage(const VkCommandBuffer&   cmdBuff,
                             size_t                   size_,
                             const void*              data_,
                             const VkImageCreateInfo& info_,
                             const VkImageLayout&     layout_ = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
  {
    ImageDedicated resultImage = createImage(info_, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    // Copy the data to staging buffer than to image
    if(data_ != nullptr)
    {
      BufferDedicated stageBuffer = createBuffer(size_, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
      m_stagingBuffers.push_back(stageBuffer);  // Remember the buffers to delete

      // Copy data to buffer
      void* mapped = nullptr;
      NVVK_CHECK(vkMapMemory(m_device, stageBuffer.allocation, 0, size_, 0, &mapped));
      memcpy(mapped, data_, size_);
      vkUnmapMemory(m_device, stageBuffer.allocation);

      // Copy buffer to image
      VkImageSubresourceRange subresourceRange{};
      subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
      subresourceRange.baseArrayLayer = 0;
      subresourceRange.baseMipLevel   = 0;
      subresourceRange.layerCount     = 1;
      subresourceRange.levelCount     = info_.mipLevels;
      nvvk::cmdBarrierImageLayout(cmdBuff, resultImage.image, VK_IMAGE_LAYOUT_UNDEFINED,
                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, subresourceRange);
      VkBufferImageCopy bufferCopyRegion{};
      bufferCopyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      bufferCopyRegion.imageSubresource.layerCount = 1;
      bufferCopyRegion.imageExtent                 = info_.extent;
      vkCmdCopyBufferToImage(cmdBuff, stageBuffer.buffer, resultImage.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &bufferCopyRegion);

      // Setting final image layout
      nvvk::cmdBarrierImageLayout(cmdBuff, resultImage.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, layout_);
    }
    else
    {
      // Setting final image layout
      nvvk::cmdBarrierImageLayout(cmdBuff, resultImage.image, VK_IMAGE_LAYOUT_UNDEFINED, layout_);
    }

    return resultImage;
  }

  //--------------------------------------------------------------------------------------------------
  // other variants could exist with a few defaults but we already have nvvk::makeImage2DViewCreateInfo()
  // we could always override viewCreateInfo.image
  TextureDedicated createTexture(const ImageDedicated& image, const VkImageViewCreateInfo& imageViewCreateInfo)
  {
    TextureDedicated resultTexture;
    resultTexture.image                  = image.image;
    resultTexture.allocation             = image.allocation;
    resultTexture.descriptor.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    assert(imageViewCreateInfo.image == image.image);
    NVVK_CHECK(vkCreateImageView(m_device, &imageViewCreateInfo, nullptr, &resultTexture.descriptor.imageView));

    return resultTexture;
  }

  TextureDedicated createTexture(const ImageDedicated&        image,
                                 const VkImageViewCreateInfo& imageViewCreateInfo,
                                 const VkSamplerCreateInfo&   samplerCreateInfo)
  {
    TextureDedicated resultTexture   = createTexture(image, imageViewCreateInfo);
    resultTexture.descriptor.sampler = m_samplerPool.acquireSampler(samplerCreateInfo);

    return resultTexture;
  }

  //--------------------------------------------------------------------------------------------------
  // shortcut that creates the image for the texture
  // - creates the image
  // - creates the texture part by associating image and sampler
  //
  TextureDedicated createTexture(const VkCommandBuffer&     cmdBuff,
                                 size_t                     size_,
                                 const void*                data_,
                                 const VkImageCreateInfo&   info_,
                                 const VkSamplerCreateInfo& samplerCreateInfo,
                                 const VkImageLayout&       layout_ = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                 bool                       isCube  = false)
  {
    ImageDedicated image = createImage(cmdBuff, size_, data_, info_, layout_);

    VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    viewInfo.pNext                           = nullptr;
    viewInfo.image                           = image.image;
    viewInfo.format                          = info_.format;
    viewInfo.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel   = 0;
    viewInfo.subresourceRange.levelCount     = VK_REMAINING_MIP_LEVELS;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount     = VK_REMAINING_ARRAY_LAYERS;
    switch(info_.imageType)
    {
      case VK_IMAGE_TYPE_1D:
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_1D;
        break;
      case VK_IMAGE_TYPE_2D:
        viewInfo.viewType = isCube ? VK_IMAGE_VIEW_TYPE_CUBE : VK_IMAGE_VIEW_TYPE_2D;
        break;
      case VK_IMAGE_TYPE_3D:
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_3D;
        break;
      default:
        assert(0);
    }

    TextureDedicated resultTexture       = createTexture(image, viewInfo, samplerCreateInfo);
    resultTexture.descriptor.imageLayout = layout_;
    return resultTexture;
  }
#ifdef VULKAN_HPP
  inline TextureDedicated createTexture(const vk::CommandBuffer&     cmdBuff,
                                        size_t                       size_,
                                        const void*                  data_,
                                        const vk::ImageCreateInfo&   info_,
                                        const vk::SamplerCreateInfo& samplerCreateInfo,
                                        const vk::ImageLayout&       layout_ = vk::ImageLayout::eShaderReadOnlyOptimal,
                                        bool                         isCube  = false)
  {
    return createTexture(static_cast<VkCommandBuffer>(cmdBuff), size_, data_, info_, samplerCreateInfo,
                         static_cast<VkImageLayout>(layout_), isCube);
  }
#endif
#if VK_NV_ray_tracing
  //--------------------------------------------------------------------------------------------------
  // Create the acceleration structure
  //
  AccelerationDedicatedNV createAcceleration(VkAccelerationStructureCreateInfoNV& accel_)
  {
    AccelerationDedicatedNV resultAccel;
    // 1. Create the acceleration structure
    NVVK_CHECK(vkCreateAccelerationStructureNV(m_device, &accel_, nullptr, &resultAccel.accel));

    // 2. Find memory requirements
    VkAccelerationStructureMemoryRequirementsInfoNV memInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_INFO_NV};
    memInfo.accelerationStructure = resultAccel.accel;
    VkMemoryRequirements2 memReqs{VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};
    vkGetAccelerationStructureMemoryRequirementsNV(m_device, &memInfo, &memReqs);


    // 3. Allocate memory
    VkMemoryAllocateInfo memAlloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    memAlloc.allocationSize = memReqs.memoryRequirements.size;
    memAlloc.memoryTypeIndex = getMemoryType(memReqs.memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    resultAccel.allocation = AllocateMemory(memAlloc);

    // 4. Bind memory with acceleration structure
    VkBindAccelerationStructureMemoryInfoNV bind{VK_STRUCTURE_TYPE_BIND_ACCELERATION_STRUCTURE_MEMORY_INFO_NV};
    bind.accelerationStructure = resultAccel.accel;
    bind.memory                = resultAccel.allocation;
    bind.memoryOffset          = 0;
    NVVK_CHECK(vkBindAccelerationStructureMemoryNV(m_device, 1, &bind));
    return resultAccel;
  }
#endif

#if VK_KHR_acceleration_structure
  //--------------------------------------------------------------------------------------------------
  // Create the acceleration structure
  //
  AccelerationDedicatedKHR createAcceleration(VkAccelerationStructureCreateInfoKHR& accel_)
  {
    AccelerationDedicatedKHR resultAccel;
    // Allocating the buffer to hold the acceleration structure
    resultAccel.buffer = createBuffer(accel_.size, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR
                                                       | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    // Setting the buffer
    accel_.buffer = resultAccel.buffer.buffer;
    // Create the acceleration structure
    vkCreateAccelerationStructureKHR(m_device, &accel_, nullptr, &resultAccel.accel);

    return resultAccel;
  }
#endif

  //--------------------------------------------------------------------------------------------------
  // Flushing staging buffers, must be done after the command buffer is submitted
  void finalizeAndReleaseStaging(VkFence fence = VkFence())
  {
    if(!m_stagingBuffers.empty())
    {
      m_garbageBuffers.push_back({fence, m_stagingBuffers});
      m_stagingBuffers.clear();
    }
    cleanGarbage();
  }

  void finalizeStaging(VkFence fence = VK_NULL_HANDLE) { finalizeAndReleaseStaging(fence); }


  //--------------------------------------------------------------------------------------------------
  // Destroy
  //
  void destroy(BufferDedicated& b_)
  {
    vkDestroyBuffer(m_device, b_.buffer, nullptr);
    vkFreeMemory(m_device, b_.allocation, nullptr);
  }

  void destroy(ImageDedicated& i_)
  {
    vkDestroyImage(m_device, i_.image, nullptr);
    vkFreeMemory(m_device, i_.allocation, nullptr);
  }

#if VK_NV_ray_tracing
  void destroy(AccelerationDedicatedNV& a_)
  {
    vkDestroyAccelerationStructureNV(m_device, a_.accel, nullptr);
    vkFreeMemory(m_device, a_.allocation, nullptr);
  }
#endif
#if VK_KHR_acceleration_structure
  void destroy(AccelerationDedicatedKHR& a_)
  {
    vkDestroyAccelerationStructureKHR(m_device, a_.accel, nullptr);
    destroy(a_.buffer);
  }
#endif


  void destroy(TextureDedicated& t_)
  {
    vkDestroyImageView(m_device, t_.descriptor.imageView, nullptr);
    vkDestroyImage(m_device, t_.image, nullptr);
    vkFreeMemory(m_device, t_.allocation, nullptr);

    if(t_.descriptor.sampler)
    {
      m_samplerPool.releaseSampler(t_.descriptor.sampler);
    }

    t_ = TextureDedicated();
  }

  //--------------------------------------------------------------------------------------------------
  // Other
  //
  void* map(const BufferDedicated& buffer_)
  {
    void* pData;
    NVVK_CHECK(vkMapMemory(m_device, buffer_.allocation, 0, VK_WHOLE_SIZE, 0, &pData));
    return pData;
  }
  void unmap(const BufferDedicated& buffer_) { vkUnmapMemory(m_device, buffer_.allocation); }


protected:
  // This is to allow Exportable Memory
  virtual VkDeviceMemory AllocateMemory(VkMemoryAllocateInfo& allocateInfo)
  {
    VkDeviceMemory mem;
    NVVK_CHECK(vkAllocateMemory(m_device, &allocateInfo, nullptr, &mem));
    return mem;
  }

  virtual void CreateBufferEx(const VkBufferCreateInfo& info_, VkBuffer* buffer)
  {
    NVVK_CHECK(vkCreateBuffer(m_device, &info_, nullptr, buffer));
  }

  virtual void CreateImageEx(const VkImageCreateInfo& info_, VkImage* image)
  {
    NVVK_CHECK(vkCreateImage(m_device, &info_, nullptr, image));
  }

  void checkMemory(const VkDeviceMemory& memory)
  {
    // If there is a leak in a DeviceMemory allocation, set the ID here to catch the object
    assert(uintptr_t(memory) != 0x00);
  }


  //--------------------------------------------------------------------------------------------------
  // Finding the memory type for memory allocation
  //
  uint32_t getMemoryType(uint32_t typeBits, const VkMemoryPropertyFlags& properties)
  {
    for(uint32_t i = 0; i < m_memoryProperties.memoryTypeCount; i++)
    {
      if(((typeBits & (1 << i)) > 0) && (m_memoryProperties.memoryTypes[i].propertyFlags & properties) == properties)
      {
        return i;
      }
    }
    assert(0);
    return ~0u;
  }

  // Clean all staging buffers, only if the associated fence is set to ready
  void cleanGarbage()
  {
    auto s = m_garbageBuffers.begin();  // Loop over all garbage
    while(s != m_garbageBuffers.end())
    {
      VkResult result = VK_SUCCESS;
      if(s->fence)  // Could be that no fence was set
      {
        result = vkGetFenceStatus(m_device, s->fence);
      }
      if(result == VK_SUCCESS)
      {
        for(auto& st : s->stagingBuffers)
        {  // Delete all buffers and free up memory
          vkDestroyBuffer(m_device, st.buffer, nullptr);
          vkFreeMemory(m_device, st.allocation, nullptr);
        }
        s = m_garbageBuffers.erase(s);  // Done with it
      }
      else
      {
        ++s;
      }
    }
  }


  struct GarbageCollection
  {
    VkFence                      fence;
    std::vector<BufferDedicated> stagingBuffers;
  };
  std::vector<GarbageCollection> m_garbageBuffers;


  VkDevice                         m_device{VK_NULL_HANDLE};
  VkPhysicalDevice                 m_physicalDevice{VK_NULL_HANDLE};
  VkPhysicalDeviceMemoryProperties m_memoryProperties{};
  std::vector<BufferDedicated>     m_stagingBuffers;
  SamplerPool                      m_samplerPool;


#ifdef VULKAN_HPP
public:
  virtual BufferDedicated createBuffer(const vk::BufferCreateInfo& info_, const vk::MemoryPropertyFlags memUsage_)
  {
    return createBuffer(static_cast<VkBufferCreateInfo>(info_), static_cast<VkMemoryPropertyFlags>(memUsage_));
  }

  BufferDedicated createBuffer(vk::DeviceSize size_, vk::BufferUsageFlags usage_, const vk::MemoryPropertyFlags memUsage_)
  {
    return createBuffer(static_cast<VkDeviceSize>(size_), static_cast<VkBufferUsageFlags>(usage_),
                        static_cast<VkMemoryPropertyFlags>(memUsage_));
  }

  BufferDedicated createBuffer(const vk::CommandBuffer&    cmdBuf,
                               const vk::DeviceSize&       size_,
                               const void*                 data_,
                               const vk::BufferUsageFlags& usage_,
                               vk::MemoryPropertyFlags     memUsage_ = vk::MemoryPropertyFlagBits::eDeviceLocal)
  {
    return createBuffer(static_cast<VkCommandBuffer>(cmdBuf), static_cast<VkDeviceSize>(size_), data_,
                        static_cast<VkBufferUsageFlags>(usage_), static_cast<VkMemoryPropertyFlags>(memUsage_));
  }

  template <typename T>
  BufferDedicated createBuffer(const vk::CommandBuffer&    cmdBuff,
                               const std::vector<T>&       data_,
                               const vk::BufferUsageFlags& usage_,
                               vk::MemoryPropertyFlags     memUsage_ = vk::MemoryPropertyFlagBits::eDeviceLocal)
  {
    return createBuffer(cmdBuff, sizeof(T) * data_.size(), data_.data(), usage_, memUsage_);
  }

  ImageDedicated createImage(const vk::ImageCreateInfo& info_, const vk::MemoryPropertyFlags memUsage_)
  {
    return createImage(static_cast<VkImageCreateInfo>(info_), static_cast<VkMemoryPropertyFlags>(memUsage_));
  }

  ImageDedicated createImage(const vk::CommandBuffer&   cmdBuff,
                             size_t                     size_,
                             const void*                data_,
                             const vk::ImageCreateInfo& info_,
                             const vk::ImageLayout&     layout_)
  {
    return createImage(static_cast<VkCommandBuffer>(cmdBuff), size_, data_, static_cast<VkImageCreateInfo>(info_),
                       static_cast<VkImageLayout>(layout_));
  }

#if VK_NV_ray_tracing
  AccelerationDedicatedNV createAcceleration(vk::AccelerationStructureCreateInfoNV& accel_)
  {
    return createAcceleration(static_cast<VkAccelerationStructureCreateInfoNV&>(accel_));
  }
#endif

#if VK_KHR_acceleration_structure
  AccelerationDedicatedKHR createAcceleration(vk::AccelerationStructureCreateInfoKHR& accel_)
  {
    return createAcceleration(static_cast<VkAccelerationStructureCreateInfoKHR&>(accel_));
  }
#endif
#endif


};  // namespace nvvk

//--------------------------------------------------------------------------------------------------
/**
  # class nvvk::AllocatorVkExport

  This version of the AllocatorDedicated will export all memory allocations, which can then be used by CUDA or OpenGL.
*/
class AllocatorVkExport : public AllocatorDedicated
{
protected:
  void CreateBufferEx(const VkBufferCreateInfo& info_, VkBuffer* buffer) override
  {
    VkBufferCreateInfo               info = info_;
    VkExternalMemoryBufferCreateInfo infoEx{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
#ifdef WIN32
    infoEx.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
    infoEx.handleTypes         = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif
    info.pNext = &infoEx;
    NVVK_CHECK(vkCreateBuffer(m_device, &info, nullptr, buffer));
  }

  void CreateImageEx(const VkImageCreateInfo& info_, VkImage* image) override
  {
    auto                            info = info_;
    VkExternalMemoryImageCreateInfo infoEx{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO};
#ifdef WIN32
    infoEx.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
    infoEx.handleTypes         = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif
    info.pNext = &infoEx;
    NVVK_CHECK(vkCreateImage(m_device, &info, nullptr, image));
  }


  // Override the standard allocation
  VkDeviceMemory AllocateMemory(VkMemoryAllocateInfo& allocateInfo) override
  {
    VkExportMemoryAllocateInfo memoryHandleEx{VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO};
#ifdef WIN32
    memoryHandleEx.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
    memoryHandleEx.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif
    allocateInfo.pNext = &memoryHandleEx;  // <-- Enabling Export
    VkDeviceMemory mem;
    NVVK_CHECK(vkAllocateMemory(m_device, &allocateInfo, nullptr, &mem));
    return mem;
  }
};

//--------------------------------------------------------------------------------------------------
// This class will export all memory allocations, to be used by OpenGL and Cuda Interop
//
class AllocatorExplicitDeviceMask : public AllocatorDedicated
{
public:
  // Initialization of the allocator
  void init(VkDevice device, VkPhysicalDevice physicalDevice, uint32_t deviceMask)
  {
    AllocatorDedicated::init(device, physicalDevice);
    m_deviceMask = deviceMask;
  }

protected:
  // Override the standard allocation
  VkDeviceMemory AllocateMemory(VkMemoryAllocateInfo& allocateInfo) override
  {
    VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    flags.deviceMask = m_deviceMask;
    flags.flags      = VK_MEMORY_ALLOCATE_DEVICE_MASK_BIT;


    allocateInfo.pNext = &flags;  // <-- Enabling Export
    VkDeviceMemory mem;
    NVVK_CHECK(vkAllocateMemory(m_device, &allocateInfo, nullptr, &mem));
    return mem;
  }

  // Target device (first device per default)
  uint32_t m_deviceMask{1u};
};

}  // namespace nvvk
