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

#if USEOPENGL

/**
  This file contains helpers for resource interoperability between OpenGL and Vulkan.
  they only exist if the shared_sources project is compiled with Vulkan AND OpenGL support.

  > WARNING: untested code
*/


#pragma once

#include <nvvk/images_vk.hpp>
#include <nvvk/memorymanagement_vkgl.hpp>

namespace nvvk {

// Objects
struct BufferDmaGL
{
  VkBuffer     buffer   = VK_NULL_HANDLE;
  GLuint       bufferGL = 0;
  AllocationID allocation;
};

struct ImageDmaGL
{
  VkImage      image = VK_NULL_HANDLE;
  GLuint       texGL = 0;
  AllocationID allocation;
};


//////////////////////////////////////////////////////////////////////////

/**
  # class nvkk::AllocatorDmaGL

  This utility has the same operations like nvvk::AllocatorDMA (see for more help), but
  targets interop between OpenGL and Vulkan.
  It uses nvkk::DeviceMemoryAllocatorGL to provide BufferDmaGL and ImageDmaGL utility classes that wrap an nvvk::AllocationID
  as well as the native Vulkan and OpenGL resource objects.

*/

//--------------------------------------------------------------------------------------------------
// Allocator for buffers, images using Device Memory Allocator
//
class AllocatorDmaGL
{
public:
  AllocatorDmaGL(AllocatorDmaGL const&) = delete;
  AllocatorDmaGL& operator=(AllocatorDmaGL const&) = delete;

  AllocatorDmaGL() = default;

  //--------------------------------------------------------------------------------------------------
  // Initialization of the allocator
  void init(VkDevice device, VkPhysicalDevice physicalDevice, nvvk::DeviceMemoryAllocatorGL* allocator, VkDeviceSize stagingBlockSize = NVVK_DEFAULT_STAGING_BLOCKSIZE)
  {
    m_device    = device;
    m_allocator = allocator;
    m_staging.init(device, physicalDevice, stagingBlockSize);
  }

  void deinit() { m_staging.deinit(); }

  // sets memory priority for VK_EXT_memory_priority
  void setPriority(float priority) { m_allocator->setPriority(priority); }

  //--------------------------------------------------------------------------------------------------
  // Basic buffer creation
  BufferDmaGL createBuffer(const VkBufferCreateInfo& info, VkMemoryPropertyFlags memProps = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  //--------------------------------------------------------------------------------------------------
  // Simple buffer creation
  BufferDmaGL createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags memProps = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  //--------------------------------------------------------------------------------------------------
  // Staging buffer creation, uploading data to device buffer
  BufferDmaGL createBuffer(VkCommandBuffer       cmd,
                           VkDeviceSize          size,
                           VkBufferUsageFlags    usage,
                           const void*           data     = nullptr,
                           VkMemoryPropertyFlags memProps = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  //--------------------------------------------------------------------------------------------------
  // Staging buffer creation, uploading data to device buffer
  template <typename T>
  BufferDmaGL createBuffer(VkCommandBuffer       cmd,
                           VkDeviceSize          size,
                           VkBufferUsageFlags    usage,
                           const std::vector<T>& data,
                           VkMemoryPropertyFlags memProps = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
  {
    BufferDmaGL resultBuffer = createBuffer(size, usage, memProps);
    if(data)
    {
      VkDeviceSize size = sizeof(T) * data.size();
      m_staging.cmdToBuffer(cmd, resultBuffer.buffer, 0, size, data.data());
    }

    return resultBuffer;
  }

  //--------------------------------------------------------------------------------------------------
  // Basic image creation
  ImageDmaGL createImage(const VkImageCreateInfo& info, GLenum formatGL, VkMemoryPropertyFlags memProps = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  //--------------------------------------------------------------------------------------------------
  // Create an image with data, data is assumed to be from first level & layer only
  //
  ImageDmaGL createImage(VkCommandBuffer          cmd,
                         const VkImageCreateInfo& info,
                         GLenum                   formatGL,
                         VkImageLayout            layout,
                         VkDeviceSize             size,
                         const void*              data,
                         VkMemoryPropertyFlags    memProps = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  //--------------------------------------------------------------------------------------------------
  // implicit staging operations triggered by create are managed here

  void finalizeStaging(VkFence fence = VK_NULL_HANDLE) { m_staging.finalizeResources(fence); }
  void finalizeAndReleaseStaging(VkFence fence = VK_NULL_HANDLE)
  {
    m_staging.finalizeResources(fence);
    m_staging.releaseResources();
  }
  void releaseStaging() { m_staging.releaseResources(); }

  StagingMemoryManager&       getStaging() { return m_staging; }
  const StagingMemoryManager& getStaging() const { return m_staging; }

  //--------------------------------------------------------------------------------------------------
  // Destroy
  //
  void destroy(BufferDmaGL& buffer);

  void destroy(ImageDmaGL& image);

  //--------------------------------------------------------------------------------------------------
  // Other
  //
  void* map(const BufferDmaGL& buffer) { return m_allocator->map(buffer.allocation); }
  void  unmap(const BufferDmaGL& buffer) { m_allocator->unmap(buffer.allocation); }


private:
  VkDevice                       m_device{VK_NULL_HANDLE};
  nvvk::DeviceMemoryAllocatorGL* m_allocator{nullptr};
  nvvk::StagingMemoryManager     m_staging;
};

}  // namespace nvvk
#endif
