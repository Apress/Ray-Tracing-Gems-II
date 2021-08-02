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

#include <vulkan/vulkan_core.h>

#include <vk_mem_alloc.h>  // VMA memory allocator

#include "images_vk.hpp"
#include "memorymanagement_vk.hpp"
#include "samplers_vk.hpp"


namespace nvvk {

// Objects
struct BufferVma
{
  VkBuffer      buffer     = VK_NULL_HANDLE;
  VmaAllocation allocation = nullptr;
};

struct ImageVma
{
  VkImage       image      = VK_NULL_HANDLE;
  VmaAllocation allocation = nullptr;
};

struct TextureVma
{
  VkImage               image      = VK_NULL_HANDLE;
  VmaAllocation         allocation = nullptr;
  VkDescriptorImageInfo descriptor{};
};

#if VK_NV_ray_tracing
struct AccelerationVmaNV
{
  VkAccelerationStructureNV accel{VK_NULL_HANDLE};
  VmaAllocation             allocation{nullptr};
};
#endif
#if VK_KHR_ray_tracing
struct AccelerationVmaKHR
{
  VkAccelerationStructureKHR accel{VK_NULL_HANDLE};
  VmaAllocation              allocation{nullptr};
};
#endif


//////////////////////////////////////////////////////////////////////////

/**
  # class nvvk::StagingMemoryManagerVma

  This utility class wraps the usage of [VMA](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator)
  to allocate the memory for nvvk::StagingMemoryManager

*/

class StagingMemoryManagerVma : public StagingMemoryManager
{
public:
  StagingMemoryManagerVma(VkDevice device, VkPhysicalDevice physicalDevice, VmaAllocator memAllocator, VkDeviceSize stagingBlockSize = NVVK_DEFAULT_STAGING_BLOCKSIZE)
  {
    init(device, physicalDevice, memAllocator, stagingBlockSize);
  }
  StagingMemoryManagerVma() {}

  void init(VkDevice device, VkPhysicalDevice physicalDevice, VmaAllocator memAllocator, VkDeviceSize stagingBlockSize = NVVK_DEFAULT_STAGING_BLOCKSIZE)
  {
    StagingMemoryManager::init(device, physicalDevice, stagingBlockSize);
    m_memAllocator = memAllocator;
  }

protected:
  VmaAllocator               m_memAllocator;
  std::vector<VmaAllocation> m_blockAllocs;

  VkResult allocBlockMemory(uint32_t index, VkDeviceSize size, bool toDevice, Block& block) override
  {
    VkBufferCreateInfo createInfo = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    createInfo.usage              = toDevice ? VK_BUFFER_USAGE_TRANSFER_SRC_BIT : VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    createInfo.size               = size;

    VmaAllocationCreateInfo allocInfo = {};
    allocInfo.usage                   = toDevice ? VMA_MEMORY_USAGE_CPU_TO_GPU : VMA_MEMORY_USAGE_GPU_TO_CPU;

    VkResult result = vmaCreateBuffer(m_memAllocator, &createInfo, &allocInfo, &block.buffer, &m_blockAllocs[index], nullptr);
    if(result != VK_SUCCESS)
    {
      return result;
    }

    result = vmaMapMemory(m_memAllocator, m_blockAllocs[index], (void**)&block.mapping);
    return result;
  }
  void freeBlockMemory(uint32_t index, const Block& block) override
  {
    vkDestroyBuffer(m_device, block.buffer, nullptr);
    vmaUnmapMemory(m_memAllocator, m_blockAllocs[index]);
    vmaFreeMemory(m_memAllocator, m_blockAllocs[index]);
  }

  void resizeBlocks(uint32_t num) override
  {
    if(num)
    {
      m_blockAllocs.resize(num);
    }
    else
    {
      m_blockAllocs.clear();
    }
  }
};

//////////////////////////////////////////////////////////////////////////

/**
  # class nvvk::AllocatorVma

  This utility class wraps the usage of [VMA](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator)
  as well as nvvk::StagingMemoryManager to have a simpler interface
  for handling resources with content uploads.

  See more details in description of [nvvk::AllocatorDma](#class-nvvkallocatordma).

*/

class AllocatorVma
{
public:
  AllocatorVma(AllocatorVma const&) = delete;
  AllocatorVma& operator=(AllocatorVma const&) = delete;

  AllocatorVma() = default;

  //--------------------------------------------------------------------------------------------------
  // Initialization of the allocator
  void init(VkDevice device, VkPhysicalDevice physicalDevice, VmaAllocator allocator, VkDeviceSize stagingBlockSize = NVVK_DEFAULT_STAGING_BLOCKSIZE)
  {
    m_device    = device;
    m_allocator = allocator;
    m_staging.init(device, physicalDevice, allocator, stagingBlockSize);
    m_samplerPool.init(device);
  }

  void deinit()
  {
    m_samplerPool.deinit();
    m_staging.deinit();
  }


  //--------------------------------------------------------------------------------------------------
  // Converter utility from Vulkan memory property to VMA
  //
  VmaMemoryUsage vkToVmaMemoryUsage(VkMemoryPropertyFlags flags)
  {
    if((flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) == VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
      return VMA_MEMORY_USAGE_GPU_ONLY;
    else if((flags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))
            == (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))
      return VMA_MEMORY_USAGE_CPU_ONLY;
    else if((flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
      return VMA_MEMORY_USAGE_CPU_TO_GPU;

    return VMA_MEMORY_USAGE_UNKNOWN;
  }

  //--------------------------------------------------------------------------------------------------
  // Basic buffer creation
  BufferVma createBuffer(const VkBufferCreateInfo& info, VmaMemoryUsage memUsage = VMA_MEMORY_USAGE_GPU_ONLY)
  {
    BufferVma               resultBuffer;
    VmaAllocationCreateInfo allocInfo = {};
    allocInfo.usage                   = memUsage;
    VkResult result = vmaCreateBuffer(m_allocator, &info, &allocInfo, &resultBuffer.buffer, &resultBuffer.allocation, nullptr);
    assert(result == VK_SUCCESS);
    return resultBuffer;
  }
  // Vulkan version
  BufferVma createBuffer(const VkBufferCreateInfo& info, const VkMemoryPropertyFlags memProps)
  {
    return createBuffer(info, vkToVmaMemoryUsage(memProps));
  }


  //--------------------------------------------------------------------------------------------------
  // Simple buffer creation
  BufferVma createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VmaMemoryUsage memUsage = VMA_MEMORY_USAGE_GPU_ONLY)
  {
    BufferVma          resultBuffer;
    VkBufferCreateInfo createInfo = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    createInfo.size               = size;
    createInfo.usage              = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    return createBuffer(createInfo, memUsage);
  }
  // Vulkan version
  BufferVma createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags memProps)
  {
    return createBuffer(size, usage, vkToVmaMemoryUsage(memProps));
  }

  //--------------------------------------------------------------------------------------------------
  // Staging buffer creation, uploading data to device buffer
  BufferVma createBuffer(VkCommandBuffer cmd, VkDeviceSize size, const void* data, VkBufferUsageFlags usage, VmaMemoryUsage memUsage = VMA_MEMORY_USAGE_GPU_ONLY)
  {
    BufferVma resultBuffer = createBuffer(size, usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT, memUsage);
    if(data)
    {
      m_staging.cmdToBuffer(cmd, resultBuffer.buffer, 0, size, data);
    }

    return resultBuffer;
  }
  // Vulkan version
  BufferVma createBuffer(VkCommandBuffer cmd, VkDeviceSize size, const void* data, VkBufferUsageFlags usage, VkMemoryPropertyFlags memProps)
  {
    return createBuffer(cmd, size, data, usage, vkToVmaMemoryUsage(memProps));
  }

  //--------------------------------------------------------------------------------------------------
  // Staging buffer creation, uploading data to device buffer
  template <typename T>
  BufferVma createBuffer(VkCommandBuffer cmd, const std::vector<T>& data, VkBufferUsageFlags usage, VmaMemoryUsage memUsage = VMA_MEMORY_USAGE_GPU_ONLY)
  {
    VkDeviceSize size         = sizeof(T) * data.size();
    BufferVma    resultBuffer = createBuffer(size, usage, memUsage);
    if(!data.empty())
    {
      m_staging.cmdToBuffer(cmd, resultBuffer.buffer, 0, size, data.data());
    }

    return resultBuffer;
  }
  // Vulkan version
  template <typename T>
  BufferVma createBuffer(VkCommandBuffer cmd, const std::vector<T>& data, VkBufferUsageFlags usage, VkMemoryPropertyFlags memProps)
  {
    return createBuffer(cmd, data, usage, vkToVmaMemoryUsage(memProps));
  }

  //--------------------------------------------------------------------------------------------------
  // Basic image creation
  ImageVma createImage(const VkImageCreateInfo& info, VmaMemoryUsage memUsage = VMA_MEMORY_USAGE_GPU_ONLY)
  {
    ImageVma                resultImage;
    VmaAllocationCreateInfo allocInfo = {};
    allocInfo.usage                   = memUsage;
    VkResult result = vmaCreateImage(m_allocator, &info, &allocInfo, &resultImage.image, &resultImage.allocation, nullptr);
    assert(result == VK_SUCCESS);
    return resultImage;
  }

  //--------------------------------------------------------------------------------------------------
  // Create an image with data, data is assumed to be from first level & layer only
  //
  ImageVma createImage(VkCommandBuffer          cmd,
                       VkDeviceSize             size,
                       const void*              data,
                       const VkImageCreateInfo& info,
                       VkImageLayout            layout   = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                       VmaMemoryUsage           memUsage = VMA_MEMORY_USAGE_GPU_ONLY)
  {
    ImageVma resultImage = createImage(info, memUsage);

    // Copy the data to staging buffer than to image
    if(data != nullptr)
    {
      // Copy buffer to image
      VkImageSubresourceRange subresourceRange{};
      subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
      subresourceRange.baseArrayLayer = 0;
      subresourceRange.baseMipLevel   = 0;
      subresourceRange.layerCount     = 1;
      subresourceRange.levelCount     = info.mipLevels;

      // doing these transitions per copy is not efficient, should do in bulk for many images
      nvvk::cmdBarrierImageLayout(cmd, resultImage.image, VK_IMAGE_LAYOUT_UNDEFINED,
                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, subresourceRange);

      VkOffset3D               offset      = {0};
      VkImageSubresourceLayers subresource = {0};
      subresource.aspectMask               = VK_IMAGE_ASPECT_COLOR_BIT;
      subresource.layerCount               = 1;

      m_staging.cmdToImage(cmd, resultImage.image, offset, info.extent, subresource, size, data);

      // Setting final image layout
      nvvk::cmdBarrierImageLayout(cmd, resultImage.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, layout);
    }
    else
    {
      // Setting final image layout
      nvvk::cmdBarrierImageLayout(cmd, resultImage.image, VK_IMAGE_LAYOUT_UNDEFINED, layout);
    }

    return resultImage;
  }


  //--------------------------------------------------------------------------------------------------
  // other variants could exist with a few defaults but we already have nvvk::makeImage2DViewCreateInfo()
  // we could always override viewCreateInfo.image
  TextureVma createTexture(const ImageVma& image, const VkImageViewCreateInfo& imageViewCreateInfo)
  {
    TextureVma resultTexture;
    resultTexture.image                  = image.image;
    resultTexture.allocation             = image.allocation;
    resultTexture.descriptor.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    assert(imageViewCreateInfo.image == image.image);
    vkCreateImageView(m_device, &imageViewCreateInfo, nullptr, &resultTexture.descriptor.imageView);

    return resultTexture;
  }

  TextureVma createTexture(const ImageVma& image, const VkImageViewCreateInfo& imageViewCreateInfo, const VkSamplerCreateInfo& samplerCreateInfo)
  {
    TextureVma resultTexture         = createTexture(image, imageViewCreateInfo);
    resultTexture.descriptor.sampler = m_samplerPool.acquireSampler(samplerCreateInfo);

    return resultTexture;
  }

  //--------------------------------------------------------------------------------------------------
  // shortcut that creates the image for the texture
  // - creates the image
  // - creates the texture part by associating image and sampler
  //
  TextureVma createTexture(const VkCommandBuffer&     cmdBuff,
                                 size_t                     size_,
                                 const void*                data_,
                                 const VkImageCreateInfo&   info_,
                                 const VkSamplerCreateInfo& samplerCreateInfo,
                                 const VkImageLayout&       layout_ = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                 bool                       isCube  = false)
  {
    ImageVma image = createImage(cmdBuff, size_, data_, info_, layout_);

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

    TextureVma resultTexture             = createTexture(image, viewInfo, samplerCreateInfo);
    resultTexture.descriptor.imageLayout = layout_;
    return resultTexture;
  }
#ifdef VULKAN_HPP
  inline TextureVma createTexture(const vk::CommandBuffer&     cmdBuff,
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
  AccelerationVmaNV createAcceleration(VkAccelerationStructureCreateInfoNV& createInfo, VmaMemoryUsage memUsage = VMA_MEMORY_USAGE_GPU_ONLY)
  {
    AccelerationVmaNV resultAccel;
    VkResult          result;

    VkAccelerationStructureNV accel;
    result = vkCreateAccelerationStructureNV(m_device, &createInfo, nullptr, &accel);
    if(result != VK_SUCCESS)
    {
      return resultAccel;
    }

    VkMemoryRequirements2                           memReqs = {VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};
    VkAccelerationStructureMemoryRequirementsInfoNV memInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_INFO_NV};
    memInfo.accelerationStructure = accel;
    vkGetAccelerationStructureMemoryRequirementsNV(m_device, &memInfo, &memReqs);

    VmaAllocation           allocation = nullptr;
    VmaAllocationInfo       allocationDetail;
    VmaAllocationCreateInfo allocInfo = {};
    allocInfo.usage                   = memUsage;
    result = vmaAllocateMemory(m_allocator, &memReqs.memoryRequirements, &allocInfo, &allocation, &allocationDetail);

    if(result != VK_SUCCESS)
    {
      vkDestroyAccelerationStructureNV(m_device, accel, nullptr);
      return resultAccel;
    }

    VkBindAccelerationStructureMemoryInfoNV bind = {VK_STRUCTURE_TYPE_BIND_ACCELERATION_STRUCTURE_MEMORY_INFO_NV};
    bind.accelerationStructure                   = accel;
    bind.memory                                  = allocationDetail.deviceMemory;
    bind.memoryOffset                            = allocationDetail.offset;

    assert(allocationDetail.offset % memReqs.memoryRequirements.alignment == 0);

    result = vkBindAccelerationStructureMemoryNV(m_device, 1, &bind);
    if(result != VK_SUCCESS)
    {
      vkDestroyAccelerationStructureNV(m_device, accel, nullptr);
      vmaFreeMemory(m_allocator, allocation);
      return resultAccel;
    }

    resultAccel.accel      = accel;
    resultAccel.allocation = allocation;
    return resultAccel;
  }
#endif
#if VK_KHR_ray_tracing

  //--------------------------------------------------------------------------------------------------
  // Create the acceleration structure
  //
  AccelerationVmaKHR createAcceleration(VkAccelerationStructureCreateInfoKHR& createInfo, VmaMemoryUsage memUsage = VMA_MEMORY_USAGE_GPU_ONLY)
  {
    AccelerationVmaKHR resultAccel;
    VkResult           result;

    VkAccelerationStructureNV accel;
    result = vkCreateAccelerationStructureKHR(m_device, &createInfo, nullptr, &accel);
    if(result != VK_SUCCESS)
    {
      return resultAccel;
    }

    VkMemoryRequirements2                            memReqs = {VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};
    VkAccelerationStructureMemoryRequirementsInfoKHR memInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_INFO_KHR};
    memInfo.accelerationStructure = accel;
    vkGetAccelerationStructureMemoryRequirementsKHR(m_device, &memInfo, &memReqs);

    VmaAllocation           allocation = nullptr;
    VmaAllocationInfo       allocationDetail;
    VmaAllocationCreateInfo allocInfo = {};
    allocInfo.usage                   = memUsage;
    result = vmaAllocateMemory(m_allocator, &memReqs.memoryRequirements, &allocInfo, &allocation, &allocationDetail);

    if(result != VK_SUCCESS)
    {
      vkDestroyAccelerationStructureKHR(m_device, accel, nullptr);
      return resultAccel;
    }

    VkBindAccelerationStructureMemoryInfoKHR bind = {VK_STRUCTURE_TYPE_BIND_ACCELERATION_STRUCTURE_MEMORY_INFO_KHR};
    bind.accelerationStructure                    = accel;
    bind.memory                                   = allocationDetail.deviceMemory;
    bind.memoryOffset                             = allocationDetail.offset;

    assert(allocationDetail.offset % memReqs.memoryRequirements.alignment == 0);

    result = vkBindAccelerationStructureMemoryKHR(m_device, 1, &bind);
    if(result != VK_SUCCESS)
    {
      vkDestroyAccelerationStructureKHR(m_device, accel, nullptr);
      vmaFreeMemory(m_allocator, allocation);
      return resultAccel;
    }

    resultAccel.accel      = accel;
    resultAccel.allocation = allocation;
    return resultAccel;
  }
#endif


  //--------------------------------------------------------------------------------------------------
  // implicit staging operations triggered by create are managed here

  void finalizeStaging(VkFence fence = VK_NULL_HANDLE) { m_staging.finalizeResources(fence); }
  void finalizeAndReleaseStaging(VkFence fence = VK_NULL_HANDLE)
  {
    m_staging.finalizeResources(fence);
    m_staging.releaseResources();
  }
  void releaseStaging() { m_staging.releaseResources(); }

  StagingMemoryManager*       getStaging() { return &m_staging; }
  const StagingMemoryManager* getStaging() const { return &m_staging; }

  //--------------------------------------------------------------------------------------------------
  // Destroy
  //
  void destroy(BufferVma& buffer)
  {
    if(buffer.buffer)
    {
      vkDestroyBuffer(m_device, buffer.buffer, nullptr);
    }
    if(buffer.allocation)
    {
      vmaFreeMemory(m_allocator, buffer.allocation);
    }

    buffer = BufferVma();
  }

  void destroy(ImageVma& image)
  {
    if(image.image)
    {
      vkDestroyImage(m_device, image.image, nullptr);
    }
    if(image.allocation)
    {
      vmaFreeMemory(m_allocator, image.allocation);
    }

    image = ImageVma();
  }

  void destroy(TextureVma& t_)
  {
    vkDestroyImageView(m_device, t_.descriptor.imageView, nullptr);
    vkDestroyImage(m_device, t_.image, nullptr);

    if(t_.descriptor.sampler)
    {
      m_samplerPool.releaseSampler(t_.descriptor.sampler);
    }
    if(t_.allocation)
    {
      vmaFreeMemory(m_allocator, t_.allocation);
    }

    t_ = TextureVma();
  }

#if VK_NV_ray_tracing
  void destroy(AccelerationVmaNV& accel)
  {
    if(accel.accel)
    {
      vkDestroyAccelerationStructureNV(m_device, accel.accel, nullptr);
    }
    if(accel.allocation)
    {
      vmaFreeMemory(m_allocator, accel.allocation);
    }

    accel = AccelerationVmaNV();
  }
#endif
#if VK_KHR_ray_tracing
  void destroy(AccelerationVmaKHR& accel)
  {
    if(accel.accel)
    {
      vkDestroyAccelerationStructureKHR(m_device, accel.accel, nullptr);
    }
    if(accel.allocation)
    {
      vmaFreeMemory(m_allocator, accel.allocation);
    }

    accel = AccelerationVmaKHR();
  }
#endif

  //--------------------------------------------------------------------------------------------------
  // Other
  //
  void* map(const BufferVma& buffer)
  {
    void* mapped;
    vmaMapMemory(m_allocator, buffer.allocation, &mapped);
    return mapped;
  }
  void unmap(const BufferVma& buffer) { vmaUnmapMemory(m_allocator, buffer.allocation); }


private:
  VkDevice                      m_device{VK_NULL_HANDLE};
  VmaAllocator                  m_allocator;
  nvvk::StagingMemoryManagerVma m_staging;
  nvvk::SamplerPool             m_samplerPool;

#ifdef VULKAN_HPP
public:
  virtual BufferVma createBuffer(const vk::BufferCreateInfo& info_, vk::MemoryPropertyFlags memUsage_)
  {
    return createBuffer(static_cast<VkBufferCreateInfo>(info_), vkToVmaMemoryUsage(static_cast<VkMemoryPropertyFlags>(memUsage_)));
  }

  BufferVma createBuffer(vk::DeviceSize size_, vk::BufferUsageFlags usage_, vk::MemoryPropertyFlags memUsage_)
  {
    return createBuffer(static_cast<VkDeviceSize>(size_), static_cast<VkBufferUsageFlags>(usage_),
                        vkToVmaMemoryUsage(static_cast<VkMemoryPropertyFlags>(memUsage_)));
  }

  BufferVma createBuffer(const vk::CommandBuffer&    cmdBuf,
                         const vk::DeviceSize&       size_,
                         const void*                 data_,
                         const vk::BufferUsageFlags& usage_,
                         vk::MemoryPropertyFlags     memUsage_)
  {
    return createBuffer(static_cast<VkCommandBuffer>(cmdBuf), static_cast<VkDeviceSize>(size_), data_,
                        static_cast<VkBufferUsageFlags>(usage_), static_cast<VkMemoryPropertyFlags>(memUsage_));
  }

  template <typename T>
  BufferVma createBuffer(const vk::CommandBuffer& cmdBuff, const std::vector<T>& data_, const vk::BufferUsageFlags& usage_, vk::MemoryPropertyFlags memUsage_)
  {
    return createBuffer(cmdBuff, sizeof(T) * data_.size(), data_.data(), usage_, memUsage_);
  }

  ImageVma createImage(const vk::ImageCreateInfo& info_, const vk::MemoryPropertyFlags memUsage_)
  {
    return createImage(static_cast<VkImageCreateInfo>(info_), vkToVmaMemoryUsage(static_cast<VkMemoryPropertyFlags>(memUsage_)));
  }

  ImageVma createImage(const vk::CommandBuffer& cmdBuff, size_t size_, const void* data_, const vk::ImageCreateInfo& info_, const vk::ImageLayout& layout_)
  {
    return createImage(static_cast<VkCommandBuffer>(cmdBuff), size_, data_, static_cast<VkImageCreateInfo>(info_),
                       static_cast<VkImageLayout>(layout_));
  }
#endif
};

}  // namespace nvvk
