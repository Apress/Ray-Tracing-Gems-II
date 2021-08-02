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

#include "images_vk.hpp"
#include "memorymanagement_vk.hpp"
#include "nvvk/error_vk.hpp"
#include "samplers_vk.hpp"

#include <memory>


/**
  # class nvvk::AllocatorDma

  The goal of the `AllocatorABC` classes is to have common work-flow
  even if the underlying allocator classes are different.
  This should make it relatively easy to switch between different
  allocator implementations (more or less only changing typedefs).

  The `BufferABC`, `ImageABC` etc. structs always contain the native
  resource handle as well as the allocator system's handle.

  This utility class wraps the usage of nvvk::DeviceMemoryAllocator
  as well as nvvk::StagingMemoryManagerDma to have a simpler interface
  for handling resources with content uploads.

  > Note: These classes are foremost to showcase principle components that
  > a Vulkan engine would most likely have.
  > They are geared towards ease of use in this sample framework, and 
  > not optimized nor meant for production code.

  ~~~ C++
  DeviceMemoryAllocator memAllocator;
  AllocatorDma          allocator;

  memAllocator.init(device, physicalDevice);
  allocator.init(device, physicalDevice, &memAllocator);

  ...

  VkCommandBuffer cmd = ... transfer queue command buffer

  // creates new resources and 
  // implicitly triggers staging transfer copy operations into cmd
  BufferDma vbo = allocator.createBuffer(cmd, vboSize, vboData, vboUsage);
  BufferDma ibo = allocator.createBuffer(cmd, iboSize, iboData, iboUsage);

  // use functions from staging memory manager
  // here we associate the temporary staging resources with a fence
  allocator.finalizeStaging( fence );

  // submit cmd buffer with staging copy operations
  vkQueueSubmit(... cmd ... fence ...)

  ...

  // if you do async uploads you would
  // trigger garbage collection somewhere per frame
  allocator.releaseStaging();

  ~~~
*/

namespace nvvk {

// Objects
struct BufferDma
{
  VkBuffer     buffer = VK_NULL_HANDLE;
  AllocationID allocation;
};

struct ImageDma
{
  VkImage      image = VK_NULL_HANDLE;
  AllocationID allocation;
};

struct TextureDma
{
  VkImage               image = VK_NULL_HANDLE;
  AllocationID          allocation;
  VkDescriptorImageInfo descriptor{};
};
#if VK_NV_ray_tracing
struct AccelerationDmaNV
{
  VkAccelerationStructureNV accel = VK_NULL_HANDLE;
  AllocationID              allocation;
};
#endif
#if VK_KHR_acceleration_structure
struct AccelerationDmaKHR
{
  VkAccelerationStructureKHR accel = VK_NULL_HANDLE;
  BufferDma                  buffer;
};
#endif

//--------------------------------------------------------------------------------------------------
// Allocator for buffers, images and acceleration structure using Device Memory Allocator
//
class AllocatorDma
{
public:
  AllocatorDma(AllocatorDma const&) = delete;
  AllocatorDma& operator=(AllocatorDma const&) = delete;

  AllocatorDma() = default;

  //--------------------------------------------------------------------------------------------------
  // Initialization of the allocator
  void init(VkDevice device, VkPhysicalDevice physicalDevice, nvvk::DeviceMemoryAllocator* allocator, VkDeviceSize stagingBlockSize = NVVK_DEFAULT_STAGING_BLOCKSIZE)
  {
    m_device    = device;
    m_allocator = allocator;
    m_staging.init(allocator, stagingBlockSize);
    m_samplerPool.init(device);
  }

  void deinit()
  {
    m_samplerPool.deinit();
    m_staging.deinit();
  }

  // sets memory priority for VK_EXT_memory_priority
  float setPriority(float priority = nvvk::DeviceMemoryAllocator::DEFAULT_PRIORITY)
  {
    return m_allocator->setPriority(priority);
  }

  //--------------------------------------------------------------------------------------------------
  // Basic buffer creation
  BufferDma createBuffer(const VkBufferCreateInfo& info, const VkMemoryPropertyFlags memProps = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
  {
    BufferDma resultBuffer;
    resultBuffer.buffer = m_allocator->createBuffer(info, resultBuffer.allocation, memProps);
    return resultBuffer;
  }

  //--------------------------------------------------------------------------------------------------
  // Simple buffer creation
  BufferDma createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags memProps = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
  {
    BufferDma resultBuffer;
    resultBuffer.buffer = m_allocator->createBuffer(size, usage, resultBuffer.allocation, memProps);
    return resultBuffer;
  }

  //--------------------------------------------------------------------------------------------------
  // Staging buffer creation, uploading data to device buffer
  BufferDma createBuffer(VkCommandBuffer cmd, VkDeviceSize size, const void* data, VkBufferUsageFlags usage, VkMemoryPropertyFlags memProps = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
  {

    BufferDma resultBuffer = createBuffer(size, usage, memProps);
    if(data)
    {
      m_staging.cmdToBuffer(cmd, resultBuffer.buffer, 0, size, data);
    }

    return resultBuffer;
  }

  //--------------------------------------------------------------------------------------------------
  // Staging buffer creation, uploading data to device buffer
  template <typename T>
  BufferDma createBuffer(VkCommandBuffer cmd, const std::vector<T>& data, VkBufferUsageFlags usage, VkMemoryPropertyFlags memProps = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
  {
    VkDeviceSize size         = sizeof(T) * data.size();
    BufferDma    resultBuffer = createBuffer(size, usage, memProps);
    if(!data.empty())
    {
      VkDeviceSize size = sizeof(T) * data.size();
      m_staging.cmdToBuffer(cmd, resultBuffer.buffer, 0, size, data.data());
    }

    return resultBuffer;
  }

  //--------------------------------------------------------------------------------------------------
  // Basic image creation
  ImageDma createImage(const VkImageCreateInfo& info, VkMemoryPropertyFlags memProps = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
  {
    ImageDma resultImage;
    resultImage.image = m_allocator->createImage(info, resultImage.allocation, memProps);
    return resultImage;
  }

  //--------------------------------------------------------------------------------------------------
  // Create an image with data, data is assumed to be from first level & layer only
  //
  ImageDma createImage(VkCommandBuffer          cmd,
                       VkDeviceSize             size,
                       const void*              data,
                       const VkImageCreateInfo& info,
                       VkImageLayout            layout   = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                       VkMemoryPropertyFlags    memProps = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
  {
    ImageDma resultImage;
    resultImage.image = m_allocator->createImage(info, resultImage.allocation, memProps);

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
  TextureDma createTexture(const ImageDma& image, const VkImageViewCreateInfo& imageViewCreateInfo)
  {
    TextureDma resultTexture;
    resultTexture.image                  = image.image;
    resultTexture.allocation             = image.allocation;
    resultTexture.descriptor.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    assert(imageViewCreateInfo.image == image.image);
    NVVK_CHECK(vkCreateImageView(m_device, &imageViewCreateInfo, nullptr, &resultTexture.descriptor.imageView));

    return resultTexture;
  }

  TextureDma createTexture(const ImageDma& image, const VkImageViewCreateInfo& imageViewCreateInfo, const VkSamplerCreateInfo& samplerCreateInfo)
  {
    TextureDma resultTexture         = createTexture(image, imageViewCreateInfo);
    resultTexture.descriptor.sampler = m_samplerPool.acquireSampler(samplerCreateInfo);

    return resultTexture;
  }

  //--------------------------------------------------------------------------------------------------
  // shortcut that creates the image for the texture
  // - creates the image
  // - creates the texture part by associating image and sampler
  //
  TextureDma createTexture(const VkCommandBuffer&     cmdBuff,
                           size_t                     size_,
                           const void*                data_,
                           const VkImageCreateInfo&   info_,
                           const VkSamplerCreateInfo& samplerCreateInfo,
                           const VkImageLayout&       layout_ = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                           bool                       isCube  = false)
  {
    ImageDma image = createImage(cmdBuff, size_, data_, info_, layout_);

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

    TextureDma resultTexture             = createTexture(image, viewInfo, samplerCreateInfo);
    resultTexture.descriptor.imageLayout = layout_;
    return resultTexture;
  }
#ifdef VULKAN_HPP
  inline TextureDma createTexture(const vk::CommandBuffer&     cmdBuff,
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
  AccelerationDmaNV createAcceleration(VkAccelerationStructureCreateInfoNV& accel,
                                       VkMemoryPropertyFlags memProps = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
  {
    AccelerationDmaNV resultAccel;
    resultAccel.accel = m_allocator->createAccStructure(accel, resultAccel.allocation, memProps);

    return resultAccel;
  }
#endif
#if VK_KHR_acceleration_structure
  AccelerationDmaKHR createAcceleration(VkAccelerationStructureCreateInfoKHR& accel,
                                        VkMemoryPropertyFlags memProps = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
  {
    AccelerationDmaKHR resultAccel;

    // Creating the buffer for the acceleration structure
    VkBufferCreateInfo createBInfo = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    createBInfo.usage              = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR
                        | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    createBInfo.size          = accel.size;
    resultAccel.buffer.buffer = m_allocator->createBuffer(createBInfo, resultAccel.buffer.allocation, memProps);

    // Create the acceleration structure
    accel.buffer = resultAccel.buffer.buffer;
    vkCreateAccelerationStructureKHR(m_device, &accel, nullptr, &resultAccel.accel);

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
  void destroy(BufferDma& buffer)
  {
    if(buffer.buffer)
    {
      vkDestroyBuffer(m_device, buffer.buffer, nullptr);
    }
    if(buffer.allocation)
    {
      m_allocator->free(buffer.allocation);
    }

    buffer = BufferDma();
  }

  void destroy(ImageDma& image)
  {
    if(image.image)
    {
      vkDestroyImage(m_device, image.image, nullptr);
    }
    if(image.allocation)
    {
      m_allocator->free(image.allocation);
    }

    image = ImageDma();
  }

  void destroy(TextureDma& t_)
  {
    vkDestroyImageView(m_device, t_.descriptor.imageView, nullptr);
    vkDestroyImage(m_device, t_.image, nullptr);

    if(t_.descriptor.sampler)
    {
      m_samplerPool.releaseSampler(t_.descriptor.sampler);
    }
    if(t_.allocation)
    {
      m_allocator->free(t_.allocation);
    }

    t_ = TextureDma();
  }
#if VK_NV_ray_tracing
  void destroy(AccelerationDmaNV& accel)
  {
    if(accel.accel)
    {
      vkDestroyAccelerationStructureNV(m_device, accel.accel, nullptr);
    }
    if(accel.allocation)
    {
      m_allocator->free(accel.allocation);
    }

    accel = AccelerationDmaNV();
  }
#endif

#if VK_KHR_acceleration_structure
  void destroy(AccelerationDmaKHR& accel)
  {
    if(accel.accel)
    {
      vkDestroyAccelerationStructureKHR(m_device, accel.accel, nullptr);
    }
    destroy(accel.buffer);

    accel = AccelerationDmaKHR();
  }
#endif

  //--------------------------------------------------------------------------------------------------
  // Other
  //
  void* map(const BufferDma& buffer) { return m_allocator->map(buffer.allocation); }
  void  unmap(const BufferDma& buffer) { m_allocator->unmap(buffer.allocation); }


private:
  VkDevice                      m_device{VK_NULL_HANDLE};
  nvvk::DeviceMemoryAllocator*  m_allocator{nullptr};
  nvvk::StagingMemoryManagerDma m_staging;
  nvvk::SamplerPool             m_samplerPool;

#ifdef VULKAN_HPP
public:
  virtual BufferDma createBuffer(const vk::BufferCreateInfo& info_, const vk::MemoryPropertyFlags memUsage_)
  {
    return createBuffer(static_cast<VkBufferCreateInfo>(info_), static_cast<VkMemoryPropertyFlags>(memUsage_));
  }

  BufferDma createBuffer(vk::DeviceSize size_, vk::BufferUsageFlags usage_, const vk::MemoryPropertyFlags memUsage_)
  {
    return createBuffer(static_cast<VkDeviceSize>(size_), static_cast<VkBufferUsageFlags>(usage_),
                        static_cast<VkMemoryPropertyFlags>(memUsage_));
  }

  BufferDma createBuffer(const vk::CommandBuffer&    cmdBuf,
                         const vk::DeviceSize&       size_,
                         const void*                 data_,
                         const vk::BufferUsageFlags& usage_,
                         vk::MemoryPropertyFlags     memUsage_ = vk::MemoryPropertyFlagBits::eDeviceLocal)
  {
    return createBuffer(static_cast<VkCommandBuffer>(cmdBuf), static_cast<VkDeviceSize>(size_), data_,
                        static_cast<VkBufferUsageFlags>(usage_), static_cast<VkMemoryPropertyFlags>(memUsage_));
  }

  template <typename T>
  BufferDma createBuffer(const vk::CommandBuffer&    cmdBuff,
                         const std::vector<T>&       data_,
                         const vk::BufferUsageFlags& usage_,
                         vk::MemoryPropertyFlags     memUsage_ = vk::MemoryPropertyFlagBits::eDeviceLocal)
  {
    return createBuffer(cmdBuff, sizeof(T) * data_.size(), data_.data(), usage_, memUsage_);
  }

  ImageDma createImage(const vk::ImageCreateInfo& info_, const vk::MemoryPropertyFlags memUsage_)
  {
    return createImage(static_cast<VkImageCreateInfo>(info_), static_cast<VkMemoryPropertyFlags>(memUsage_));
  }

  ImageDma createImage(const vk::CommandBuffer& cmdBuff, size_t size_, const void* data_, const vk::ImageCreateInfo& info_, const vk::ImageLayout& layout_)
  {
    return createImage(static_cast<VkCommandBuffer>(cmdBuff), size_, data_, static_cast<VkImageCreateInfo>(info_),
                       static_cast<VkImageLayout>(layout_));
  }

#if VK_NV_ray_tracing
  AccelerationDmaNV createAcceleration(vk::AccelerationStructureCreateInfoNV& accel_)
  {
    return createAcceleration(static_cast<VkAccelerationStructureCreateInfoNV&>(accel_));
  }
#endif
#endif
};

}  // namespace nvvk
