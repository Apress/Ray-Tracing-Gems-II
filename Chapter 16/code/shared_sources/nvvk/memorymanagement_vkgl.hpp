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
*/


#pragma once

#include <nvgl/extensions_gl.hpp>
#include <nvvk/images_vk.hpp>
#include <nvvk/memorymanagement_vk.hpp>
#include <vulkan/vulkan_core.h>

namespace nvvk {

struct AllocationGL
{
  GLuint   memoryObject = 0;
  GLuint64 offset       = 0;
  GLuint64 size         = 0;
};

//////////////////////////////////////////////////////////////////////////

/** 
  # class nvvk::DeviceMemoryAllocatorGL

  Derived from nvvk::DeviceMemoryAllocator it uses vulkan memory that is exported
  and directly imported into OpenGL. Requires GL_EXT_memory_object.

  Used just like the original class however a new function to get the 
  GL memory object exists: `getAllocationGL`.

  Look at source of nvvk::AllocatorDmaGL for usage.
*/


class DeviceMemoryAllocatorGL : public DeviceMemoryAllocator
{
public:
  DeviceMemoryAllocatorGL() {}
  DeviceMemoryAllocatorGL(VkDevice         device,
                          VkPhysicalDevice physicalDevice,
                          VkDeviceSize     blockSize = NVVK_DEFAULT_MEMORY_BLOCKSIZE,
                          VkDeviceSize     maxSize   = NVVK_DEFAULT_MAX_MEMORY_ALLOCATIONSIZE)
      : DeviceMemoryAllocator(device, physicalDevice, blockSize, maxSize)
  {
  }


  AllocationGL getAllocationGL(AllocationID aid) const
  {
    AllocationGL          alloc;
    const AllocationInfo& info = getInfo(aid);
    alloc.memoryObject         = m_blockGLs[info.block.index].memoryObject;
    alloc.offset               = info.allocation.offset;
    alloc.size                 = info.allocation.size;
    return alloc;
  }

  static VkExternalMemoryHandleTypeFlags getExternalMemoryHandleTypeFlags();

protected:
  struct BlockGL
  {
#ifdef WIN32
    void* handle = nullptr;
#else
    int handle = -1;
#endif
    GLuint memoryObject = 0;
  };

  std::vector<BlockGL> m_blockGLs;

  struct StructChain
  {
    VkStructureType    sType;
    const StructChain* pNext;
  };

  VkResult allocBlockMemory(BlockID id, VkMemoryAllocateInfo& memInfo, VkDeviceMemory& deviceMemory) override;
  void     freeBlockMemory(BlockID id, VkDeviceMemory deviceMemory) override;
  void     resizeBlocks(uint32_t count) override;

  VkResult createBufferInternal(VkDevice device, const VkBufferCreateInfo* info, VkBuffer* buffer) override;
  VkResult createImageInternal(VkDevice device, const VkImageCreateInfo* info, VkImage* image) override;

};
}  // namespace nvvk
#endif
